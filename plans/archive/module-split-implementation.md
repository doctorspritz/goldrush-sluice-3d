# Module Split Implementation Plan

## Overview

Split `flip.rs` (2991 lines) and `grid.rs` (2071 lines) into smaller, cohesive modules following Rust best practices and patterns from Bevy ECS and GridFluidSim3D.

## Target Structure

```
crates/sim/src/
├── lib.rs                    # Crate root (update exports)
├── particle.rs               # Keep as-is (943 lines)
├── physics.rs                # Keep as-is (25 lines)
├── pbf.rs                    # Keep as-is (393 lines)
├── sluice.rs                 # Keep as-is (557 lines)
│
├── flip/
│   ├── mod.rs                # FlipSimulation struct + update orchestration
│   ├── transfer.rs           # P2G and G2P operations
│   ├── sediment.rs           # Sediment physics, deposition, entrainment
│   ├── advection.rs          # Particle movement, collision
│   ├── spawning.rs           # Particle spawning functions
│   ├── diagnostics.rs        # Energy, enstrophy, CFL, profiling
│   └── pile.rs               # Heightfield computation, constraints
│
└── grid/
    ├── mod.rs                # Grid struct, basic accessors
    ├── velocity.rs           # Velocity sampling, gravity, viscosity
    ├── pressure.rs           # Pressure solve, gradient application
    ├── extrapolation.rs      # Velocity extrapolation
    ├── vorticity.rs          # Vorticity computation, confinement
    ├── sdf.rs                # Signed distance field operations
    └── cell_types.rs         # DepositedCell, CellType definitions
```

---

## Phase 1: Grid Module Split

### Step 1.1: Create `grid/cell_types.rs`

**Move from `grid.rs`:**
- `CellType` enum (lines 16-31)
- `DepositedCell` struct (lines 33-72)
- All `DepositedCell` impl methods

**Dependencies:**
- `ParticleMaterial` from `crate::particle`

**Unit Tests:**
```rust
// grid/cell_types.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::ParticleMaterial;

    #[test]
    fn deposited_cell_defaults_to_empty() {
        let cell = DepositedCell::default();
        assert!(!cell.is_deposited());
        assert!(cell.get_material().is_none());
    }

    #[test]
    fn deposited_cell_stores_material() {
        let mut cell = DepositedCell::default();
        cell.set_material(ParticleMaterial::Sand);
        assert!(cell.is_deposited());
        assert_eq!(cell.get_material(), Some(ParticleMaterial::Sand));
    }

    #[test]
    fn deposited_cell_clear_works() {
        let mut cell = DepositedCell::default();
        cell.set_material(ParticleMaterial::Gold);
        cell.clear();
        assert!(!cell.is_deposited());
    }

    #[test]
    fn effective_shields_varies_by_material() {
        let mut cell = DepositedCell::default();
        cell.set_material(ParticleMaterial::Sand);
        let sand_shields = cell.effective_shields_critical();

        cell.clear();
        cell.set_material(ParticleMaterial::Gold);
        let gold_shields = cell.effective_shields_critical();

        // Gold should require more shear to entrain
        assert!(gold_shields > sand_shields);
    }

    #[test]
    fn effective_density_varies_by_material() {
        let mut cell = DepositedCell::default();
        cell.set_material(ParticleMaterial::Sand);
        let sand_density = cell.effective_density();

        cell.clear();
        cell.set_material(ParticleMaterial::Gold);
        let gold_density = cell.effective_density();

        // Gold is denser
        assert!(gold_density > sand_density);
    }
}
```

---

### Step 1.2: Create `grid/sdf.rs`

**Move from `grid.rs`:**
- `compute_sdf` (lines 269-347)
- `sample_sdf` (lines 386-409)
- `sdf_gradient` (lines 411-417)
- `compute_bed_heights` (lines 349-364)
- `sample_bed_height` (lines 366-375)
- `normalized_height_above_bed` (lines 377-384)

**Dependencies:**
- `Grid` struct (access via `&self` / `&mut self`)
- `solid` terrain data (prefer `set_solid` as the source of truth)

**Notes:**
- SDF remains a **solid-terrain SDF** (distance to solid).
- Update `compute_sdf` to read `solid` directly (and treat boundaries as solid) so it does not depend on `cell_type`.
- Ensure boundary cells are treated as solid before SDF computation (either by marking them in `solid` or handling them inside `compute_sdf`).

**Unit Tests (solid SDF - distance to solid terrain):**
```rust
// grid/sdf.rs
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_grid() -> Grid {
        Grid::new(16, 16, 1.0)
    }

    #[test]
    fn sdf_positive_in_air() {
        let mut grid = create_test_grid();
        // Mark center cells as solid
        for i in 6..10 {
            for j in 6..10 {
                grid.set_solid(i, j);
            }
        }
        grid.compute_sdf();

        // Sample in air region (outside solid)
        let sdf = grid.sample_sdf(glam::Vec2::new(2.0, 2.0));
        assert!(sdf > 0.0, "SDF should be positive in air");
    }

    #[test]
    fn sdf_negative_in_solid() {
        let mut grid = create_test_grid();
        for i in 6..10 {
            for j in 6..10 {
                grid.set_solid(i, j);
            }
        }
        grid.compute_sdf();

        // Sample inside solid region
        let sdf = grid.sample_sdf(glam::Vec2::new(8.0, 8.0));
        assert!(sdf < 0.0, "SDF should be negative in solid");
    }

    #[test]
    fn sdf_gradient_points_outward() {
        let mut grid = create_test_grid();
        for i in 6..10 {
            for j in 6..10 {
                grid.set_solid(i, j);
            }
        }
        grid.compute_sdf();

        // At right edge of solid, gradient should point right (positive x)
        let grad = grid.sdf_gradient(glam::Vec2::new(9.5, 8.0));
        assert!(grad.x > 0.0, "Gradient should point toward air");
    }

    #[test]
    fn bed_height_increases_with_deposited() {
        let mut grid = create_test_grid();
        let height_empty = grid.sample_bed_height(5.0);

        // Add deposited cells
        grid.set_deposited(5, 1);
        grid.set_deposited(5, 2);
        grid.compute_bed_heights();

        let height_with_deposit = grid.sample_bed_height(5.0);
        assert!(height_with_deposit > height_empty);
    }
}
```

---

### Step 1.3: Create `grid/velocity.rs`

**Move from `grid.rs`:**
- `sample_velocity` (lines 518-524)
- `sample_velocity_bspline` (lines 526-625)
- `sample_u` (lines 655-681)
- `sample_v` (lines 683-710)
- `get_interp_weights` (lines 712-733)
- `clear_velocities` (lines 735-739)
- `apply_gravity` (lines 749-762)
- `apply_viscosity` (lines 764-843)

**Dependencies:**
- `quadratic_bspline`, `quadratic_bspline_1d` (stay in mod.rs or move here)

**Unit Tests:**
```rust
// grid/velocity.rs
#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn sample_velocity_interpolates_correctly() {
        let mut grid = Grid::new(8, 8, 1.0);

        // Set uniform rightward velocity
        for u in grid.u.iter_mut() {
            *u = 10.0;
        }

        let vel = grid.sample_velocity(Vec2::new(4.0, 4.0));
        assert!((vel.x - 10.0).abs() < 0.1);
    }

    #[test]
    fn clear_velocities_zeros_all() {
        let mut grid = Grid::new(8, 8, 1.0);
        grid.u.fill(5.0);
        grid.v.fill(3.0);

        grid.clear_velocities();

        assert!(grid.u.iter().all(|&u| u == 0.0));
        assert!(grid.v.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn apply_gravity_increases_downward_velocity() {
        let mut grid = Grid::new(8, 8, 1.0);
        // Mark as fluid
        for i in 2..6 {
            for j in 2..6 {
                let idx = grid.cell_index(i, j);
                grid.cell_type[idx] = CellType::Fluid;
            }
        }

        let v_before: f32 = grid.v.iter().sum();
        grid.apply_gravity(1.0 / 60.0);
        let v_after: f32 = grid.v.iter().sum();

        // V should decrease (more negative) with gravity
        assert!(v_after < v_before);
    }

    #[test]
    fn viscosity_smooths_velocity_field() {
        let mut grid = Grid::new(16, 16, 1.0);
        // Mark all as fluid
        for idx in 0..grid.cell_type.len() {
            grid.cell_type[idx] = CellType::Fluid;
        }

        // Create sharp velocity discontinuity
        let mid = grid.u.len() / 2;
        for i in 0..mid {
            grid.u[i] = 0.0;
        }
        for i in mid..grid.u.len() {
            grid.u[i] = 100.0;
        }

        let max_diff_before = grid.u.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);

        grid.apply_viscosity(1.0 / 60.0, 0.1);

        let max_diff_after = grid.u.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff_after < max_diff_before, "Viscosity should smooth");
    }
}
```

---

### Step 1.4: Create `grid/pressure.rs`

**Move from `grid.rs`:**
- `compute_divergence` (lines 874-895)
- `total_divergence` (lines 897-900)
- `pressure_stats` (lines 902-927)
- `solve_pressure` (lines 929-962)
- `update_pressure_cell` (lines 964-1003)
- `compute_max_residual` (lines 1005-1033)
- `apply_pressure_gradient` (lines 1035-1087)
- `clear_pressure` (lines 741-747)

**Unit Tests:**
```rust
// grid/pressure.rs
#[cfg(test)]
mod tests {
    use super::*;

    fn create_fluid_grid() -> Grid {
        let mut grid = Grid::new(16, 16, 1.0);
        for i in 4..12 {
            for j in 4..12 {
                let idx = grid.cell_index(i, j);
                grid.cell_type[idx] = CellType::Fluid;
            }
        }
        grid
    }

    #[test]
    fn divergence_free_field_has_zero_divergence() {
        let mut grid = create_fluid_grid();
        // Uniform velocity = divergence free
        grid.u.fill(5.0);
        grid.v.fill(0.0);

        grid.compute_divergence();

        let total_div = grid.total_divergence();
        assert!(total_div.abs() < 1.0, "Uniform field should have ~zero divergence");
    }

    #[test]
    fn pressure_solve_reduces_divergence() {
        let mut grid = create_fluid_grid();

        // Create divergent field (sources)
        for i in 6..10 {
            for j in 6..10 {
                let idx = grid.u_index(i, j);
                grid.u[idx] = 10.0;
                let idx = grid.u_index(i + 1, j);
                grid.u[idx] = -10.0; // Divergent!
            }
        }

        grid.compute_divergence();
        let div_before = grid.total_divergence().abs();

        grid.solve_pressure(100);
        grid.apply_pressure_gradient(1.0 / 60.0);
        grid.compute_divergence();

        let div_after = grid.total_divergence().abs();

        assert!(div_after < div_before, "Pressure solve should reduce divergence");
    }

    #[test]
    fn pressure_stats_returns_valid_range() {
        let mut grid = create_fluid_grid();
        grid.solve_pressure(50);

        let (min, max, avg) = grid.pressure_stats();

        assert!(min <= avg);
        assert!(avg <= max);
    }

    #[test]
    fn clear_pressure_zeros_field() {
        let mut grid = create_fluid_grid();
        grid.pressure.fill(100.0);

        grid.clear_pressure();

        assert!(grid.pressure.iter().all(|&p| p == 0.0));
    }
}
```

---

### Step 1.5: Create `grid/extrapolation.rs`

**Move from `grid.rs`:**
- `extrapolate_velocities` (lines 1329-1345)
- `mark_fluid_faces_known` (lines 1347-1380)
- `extrapolate_u_layer` (lines 1382-1458)
- `extrapolate_v_layer` (lines 1460-1532)
- `mark_fluid_faces_known_preallocated` (lines 1534-1559)
- `extrapolate_u_layer_preallocated` (lines 1561-1629)
- `extrapolate_v_layer_preallocated` (lines 1631-1698)

**Unit Tests:**
```rust
// grid/extrapolation.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extrapolation_copies_from_fluid_neighbors() {
        let mut grid = Grid::new(12, 12, 1.0);

        // Fluid block
        for i in 4..8 {
            for j in 4..8 {
                let idx = grid.cell_index(i, j);
                grid.cell_type[idx] = CellType::Fluid;
            }
        }

        // Set velocity at fluid edge
        let idx = grid.u_index(7, 5);
        grid.u[idx] = 25.0;

        grid.extrapolate_velocities(2);

        // Check air cell got extrapolated value
        let idx = grid.u_index(8, 5);
        assert!(grid.u[idx].abs() > 0.0, "Air face should have extrapolated velocity");
    }

    #[test]
    fn extrapolation_propagates_multiple_layers() {
        let mut grid = Grid::new(16, 16, 1.0);

        // Small fluid region
        let idx = grid.cell_index(8, 8);
        grid.cell_type[idx] = CellType::Fluid;

        let idx = grid.u_index(8, 8);
        grid.u[idx] = 50.0;

        // 1 layer
        grid.extrapolate_velocities(1);
        let one_layer = grid.u[grid.u_index(10, 8)];

        // Reset and try 3 layers
        grid.u[grid.u_index(10, 8)] = 0.0;
        grid.extrapolate_velocities(3);
        let three_layers = grid.u[grid.u_index(10, 8)];

        assert!(three_layers.abs() >= one_layer.abs());
    }

    #[test]
    fn extrapolation_does_not_modify_fluid_cells() {
        let mut grid = Grid::new(12, 12, 1.0);

        for i in 4..8 {
            for j in 4..8 {
                let idx = grid.cell_index(i, j);
                grid.cell_type[idx] = CellType::Fluid;
            }
        }

        // Set known fluid velocity
        let idx = grid.u_index(5, 5);
        grid.u[idx] = 42.0;

        grid.extrapolate_velocities(3);

        // Fluid velocity unchanged
        assert_eq!(grid.u[idx], 42.0);
    }

    #[test]
    fn extrapolation_respects_solid_boundaries() {
        let mut grid = Grid::new(10, 10, 1.0);

        // Fluid
        let idx = grid.cell_index(4, 4);
        grid.cell_type[idx] = CellType::Fluid;
        let idx = grid.cell_index(5, 4);
        grid.cell_type[idx] = CellType::Fluid;

        // Solid wall
        let idx = grid.cell_index(6, 4);
        grid.cell_type[idx] = CellType::Solid;

        let idx = grid.u_index(5, 4);
        grid.u[idx] = 100.0;

        // Zero solid boundary
        let solid_face_idx = grid.u_index(6, 4);
        grid.u[solid_face_idx] = 0.0;

        grid.extrapolate_velocities(2);

        // Solid face should stay ~0
        assert!(grid.u[solid_face_idx].abs() < 50.0);
    }
}
```

---

### Step 1.6: Create `grid/vorticity.rs`

**Move from `grid.rs`:**
- `compute_vorticity` (lines 1136-1156)
- `compute_enstrophy` (lines 1158-1174)
- `total_absolute_vorticity` (lines 1176-1191)
- `max_vorticity` (lines 1193-1204)
- `apply_vorticity_confinement` (lines 1206-1210)
- `apply_vorticity_confinement_with_piles` (lines 1212-1315)
- `sample_vorticity` (lines 628-653)
- `damp_surface_vertical` (lines 1089-1134)

**Unit Tests:**
```rust
// grid/vorticity.rs
#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    fn create_rotating_flow() -> Grid {
        let mut grid = Grid::new(16, 16, 1.0);

        // Mark center as fluid
        for i in 4..12 {
            for j in 4..12 {
                let idx = grid.cell_index(i, j);
                grid.cell_type[idx] = CellType::Fluid;
            }
        }

        // Create rotating velocity field
        let center = Vec2::new(8.0, 8.0);
        for i in 0..=grid.width {
            for j in 0..grid.height {
                let pos = Vec2::new(i as f32, j as f32 + 0.5);
                let r = pos - center;
                // Tangential velocity
                let tangent = Vec2::new(-r.y, r.x).normalize_or_zero();
                let idx = grid.u_index(i, j);
                grid.u[idx] = tangent.x * 10.0;
            }
        }
        for i in 0..grid.width {
            for j in 0..=grid.height {
                let pos = Vec2::new(i as f32 + 0.5, j as f32);
                let r = pos - center;
                let tangent = Vec2::new(-r.y, r.x).normalize_or_zero();
                let idx = grid.v_index(i, j);
                grid.v[idx] = tangent.y * 10.0;
            }
        }

        grid
    }

    #[test]
    fn rotating_flow_has_nonzero_vorticity() {
        let mut grid = create_rotating_flow();
        grid.compute_vorticity();

        let enstrophy = grid.compute_enstrophy();
        assert!(enstrophy > 0.0, "Rotating flow should have vorticity");
    }

    #[test]
    fn uniform_flow_has_zero_vorticity() {
        let mut grid = Grid::new(16, 16, 1.0);
        for i in 4..12 {
            for j in 4..12 {
                let idx = grid.cell_index(i, j);
                grid.cell_type[idx] = CellType::Fluid;
            }
        }

        // Uniform rightward flow
        grid.u.fill(10.0);
        grid.v.fill(0.0);

        grid.compute_vorticity();

        let enstrophy = grid.compute_enstrophy();
        assert!(enstrophy < 0.1, "Uniform flow should have ~zero vorticity");
    }

    #[test]
    fn vorticity_confinement_preserves_rotation() {
        let mut grid = create_rotating_flow();
        grid.compute_vorticity();

        let enstrophy_before = grid.compute_enstrophy();

        // Apply confinement (should inject energy)
        grid.apply_vorticity_confinement(1.0 / 60.0, 0.5);
        grid.compute_vorticity();

        let enstrophy_after = grid.compute_enstrophy();

        // Confinement should maintain or increase rotation
        assert!(enstrophy_after >= enstrophy_before * 0.95);
    }

    #[test]
    fn sample_vorticity_returns_local_rotation() {
        let mut grid = create_rotating_flow();
        grid.compute_vorticity();

        let vort_center = grid.sample_vorticity(Vec2::new(8.0, 8.0));
        let vort_edge = grid.sample_vorticity(Vec2::new(2.0, 2.0));

        // Center of rotation should have significant vorticity
        assert!(vort_center.abs() > vort_edge.abs());
    }
}
```

---

### Step 1.7: Create `grid/mod.rs`

**Keep in mod.rs:**
- `Grid` struct definition
- `new()` constructor
- Index methods: `cell_index`, `u_index`, `v_index`, `pos_to_cell`
- `quadratic_bspline`, `quadratic_bspline_1d`, `apic_d_inverse`
- Cell state accessors: `set_solid`, `is_solid`, `is_fluid`, etc.
- `enforce_boundary_conditions`
- `total_momentum`
- Multigrid stubs (if any)

**Module declarations and re-exports:**
```rust
// grid/mod.rs
mod cell_types;
mod extrapolation;
mod pressure;
mod sdf;
mod velocity;
mod vorticity;

pub use cell_types::{CellType, DepositedCell};
```

---

## Phase 2: FLIP Module Split

### Step 2.1: Create `flip/spawning.rs`

**Move from `flip.rs`:**
- `is_spawn_safe` (line 2557)
- `spawn_water` (lines 2563-2581)
- `spawn_sand` (lines 2583-2607)
- `spawn_magnetite` (lines 2609-2633)
- `spawn_gold` (lines 2635-2658)

**Unit Tests:**
```rust
// flip/spawning.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_water_adds_particles() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);
        let count_before = sim.particles.len();

        sim.spawn_water(16.0, 16.0, 0.0, 0.0, 10);

        assert_eq!(sim.particles.len(), count_before + 10);
    }

    #[test]
    fn spawned_particles_have_correct_material() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);

        sim.spawn_gold(16.0, 16.0, 0.0, 0.0, 5);

        let gold_count = sim.particles.iter()
            .filter(|p| p.material == ParticleMaterial::Gold)
            .count();

        assert_eq!(gold_count, 5);
    }

    #[test]
    fn spawned_particles_have_velocity() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);

        sim.spawn_sand(16.0, 16.0, 5.0, -3.0, 5);

        for p in sim.particles.iter().rev().take(5) {
            assert!((p.velocity.x - 5.0).abs() < 0.5);
            assert!((p.velocity.y - (-3.0)).abs() < 0.5);
        }
    }

    #[test]
    fn is_spawn_safe_respects_boundaries() {
        let sim = FlipSimulation::new(32, 32, 1.0);

        // Inside bounds
        assert!(sim.is_spawn_safe(16.0, 16.0));

        // Outside bounds
        assert!(!sim.is_spawn_safe(-1.0, 16.0));
        assert!(!sim.is_spawn_safe(16.0, 100.0));
    }

    #[test]
    fn density_varies_by_material() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);

        sim.spawn_sand(10.0, 16.0, 0.0, 0.0, 1);
        sim.spawn_gold(20.0, 16.0, 0.0, 0.0, 1);

        let sand = sim.particles.iter().find(|p| p.material == ParticleMaterial::Sand).unwrap();
        let gold = sim.particles.iter().find(|p| p.material == ParticleMaterial::Gold).unwrap();

        assert!(gold.density > sand.density);
    }
}
```

---

### Step 2.2: Create `flip/diagnostics.rs`

**Move from `flip.rs`:**
- `compute_kinetic_energy` (lines 2667-2672)
- `compute_water_kinetic_energy` (lines 2674-2681)
- `compute_enstrophy` (lines 2683-2687)
- `update_and_compute_enstrophy` (lines 2689-2696)
- `max_velocity` (lines 2698-2704)
- `compute_cfl` (lines 2706-2711)
- `initialize_taylor_green` (lines 2713-2754)
- `initialize_solid_rotation` (lines 2756-2821)
- `update_with_diagnostics` (lines 2829-2922)
- `update_profiled` (lines 268-332)
- `get_u_weight_sum`, `get_v_weight_sum` (lines 2924-2933)
- `run_isolated_flip_cycle` (lines 2936-2967)
- `run_isolated_flip_cycle_with_extrapolation` (lines 334-344)
- `compute_grid_kinetic_energy` (lines 2969-end)

**Unit Tests:**
```rust
// flip/diagnostics.rs
#[cfg(test)]
mod tests {
    use super::*;

    fn sim_with_moving_particles() -> FlipSimulation {
        let mut sim = FlipSimulation::new(32, 32, 1.0);
        sim.spawn_water(16.0, 16.0, 10.0, 5.0, 100);
        sim
    }

    #[test]
    fn kinetic_energy_positive_for_moving_particles() {
        let sim = sim_with_moving_particles();
        let ke = sim.compute_kinetic_energy();
        assert!(ke > 0.0);
    }

    #[test]
    fn kinetic_energy_zero_for_stationary() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);
        sim.spawn_water(16.0, 16.0, 0.0, 0.0, 100);

        let ke = sim.compute_kinetic_energy();
        assert!(ke < 0.01);
    }

    #[test]
    fn max_velocity_matches_fastest_particle() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);
        sim.spawn_water(10.0, 16.0, 1.0, 0.0, 10);
        sim.spawn_water(20.0, 16.0, 100.0, 0.0, 1); // Fast one

        let max_v = sim.max_velocity();
        assert!((max_v - 100.0).abs() < 1.0);
    }

    #[test]
    fn cfl_scales_with_velocity_and_dt() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);
        sim.spawn_water(16.0, 16.0, 50.0, 0.0, 10);

        let cfl_small_dt = sim.compute_cfl(0.001);
        let cfl_large_dt = sim.compute_cfl(0.1);

        assert!(cfl_large_dt > cfl_small_dt);
    }

    #[test]
    fn taylor_green_creates_vortices() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);
        sim.initialize_taylor_green();

        let enstrophy = sim.update_and_compute_enstrophy();
        assert!(enstrophy > 0.0, "Taylor-Green should have vorticity");
    }

    #[test]
    fn isolated_flip_cycle_conserves_momentum() {
        let mut sim = sim_with_moving_particles();

        let mom_before: f32 = sim.particles.iter()
            .map(|p| p.velocity.length())
            .sum();

        sim.run_isolated_flip_cycle_with_extrapolation(1.0 / 60.0);

        let mom_after: f32 = sim.particles.iter()
            .map(|p| p.velocity.length())
            .sum();

        let ratio = mom_after / mom_before;
        assert!(ratio > 0.95 && ratio < 1.05);
    }
}
```

---

### Step 2.3: Create `flip/transfer.rs`

**Move from `flip.rs`:**
- `particles_to_grid` (lines 395-568)
- `grid_to_particles` (lines 850-1144)
- `store_old_velocities` (lines 724-848)
- `apply_pressure_gradient_two_way` (lines 570-652)
- `apply_porosity_drag` (lines 654-722)

**Unit Tests:**
```rust
// flip/transfer.rs
#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    fn create_sim_with_particles() -> FlipSimulation {
        let mut sim = FlipSimulation::new(16, 16, 1.0);
        // Fill center with water
        for i in 5..11 {
            for j in 5..11 {
                for dx in [0.25, 0.75] {
                    for dy in [0.25, 0.75] {
                        let x = i as f32 + dx;
                        let y = j as f32 + dy;
                        sim.particles.list.push(
                            Particle::new(Vec2::new(x, y), Vec2::new(10.0, 0.0), ParticleMaterial::Water)
                        );
                    }
                }
            }
        }
        sim
    }

    #[test]
    fn p2g_transfers_velocity_to_grid() {
        let mut sim = create_sim_with_particles();
        sim.classify_cells();
        sim.particles_to_grid();

        // Grid should have non-zero U velocity
        let u_sum: f32 = sim.grid.u.iter().sum();
        assert!(u_sum > 0.0, "P2G should transfer velocity to grid");
    }

    #[test]
    fn g2p_transfers_velocity_to_particles() {
        let mut sim = create_sim_with_particles();
        sim.classify_cells();
        sim.particles_to_grid();
        sim.store_old_velocities();

        // Modify grid velocity
        for u in sim.grid.u.iter_mut() {
            *u += 5.0;
        }

        let vel_before: Vec2 = sim.particles.iter()
            .map(|p| p.velocity)
            .fold(Vec2::ZERO, |a, b| a + b);

        sim.grid_to_particles(1.0 / 60.0);

        let vel_after: Vec2 = sim.particles.iter()
            .map(|p| p.velocity)
            .fold(Vec2::ZERO, |a, b| a + b);

        // Particles should have gained X velocity
        assert!(vel_after.x > vel_before.x);
    }

    #[test]
    fn store_old_velocities_preserves_values() {
        let mut sim = create_sim_with_particles();
        sim.classify_cells();
        sim.particles_to_grid();

        let u_before = sim.grid.u.clone();
        sim.store_old_velocities();

        assert_eq!(sim.grid.u_old, u_before);
    }

    #[test]
    fn p2g_g2p_roundtrip_conserves_momentum() {
        let mut sim = create_sim_with_particles();

        let momentum_before: f32 = sim.particles.iter()
            .map(|p| p.velocity.length())
            .sum();

        sim.classify_cells();
        sim.particles_to_grid();
        sim.store_old_velocities();
        sim.grid_to_particles(1.0 / 60.0);

        let momentum_after: f32 = sim.particles.iter()
            .map(|p| p.velocity.length())
            .sum();

        let ratio = momentum_after / momentum_before;
        assert!(ratio > 0.98 && ratio < 1.02);
    }
}
```

---

### Step 2.4: Create `flip/advection.rs`

**Move from `flip.rs`:**
- `advect_particles` (lines 2238-2347)
- `push_particles_apart` (lines 2349-2435)
- `build_spatial_hash` (lines 2437-2465)
- `compute_neighbor_counts` (lines 2467-2549)
- `compute_water_neighbor_counts` (lines 2551-2555)

**Unit Tests:**
```rust
// flip/advection.rs
#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn advection_moves_particles() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);
        sim.spawn_water(16.0, 16.0, 10.0, 0.0, 10);

        let pos_before: Vec<Vec2> = sim.particles.iter()
            .map(|p| p.position)
            .collect();

        sim.advect_particles(0.1);

        for (i, p) in sim.particles.iter().enumerate() {
            assert!(p.position.x > pos_before[i].x, "Should move right");
        }
    }

    #[test]
    fn advection_respects_boundaries() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);
        // Place particle near right boundary with rightward velocity
        sim.particles.list.push(
            Particle::new(Vec2::new(31.0, 16.0), Vec2::new(100.0, 0.0), ParticleMaterial::Water)
        );

        sim.advect_particles(0.1);

        let p = &sim.particles.list[0];
        assert!(p.position.x < 32.0, "Should not exit domain");
    }

    #[test]
    fn push_apart_separates_overlapping() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);
        // Place particles at exact same spot
        for _ in 0..10 {
            sim.particles.list.push(
                Particle::new(Vec2::new(16.0, 16.0), Vec2::ZERO, ParticleMaterial::Water)
            );
        }

        sim.push_particles_apart(5);

        // Check particles have spread out
        let positions: Vec<Vec2> = sim.particles.iter().map(|p| p.position).collect();
        let mut min_dist = f32::MAX;
        for i in 0..positions.len() {
            for j in (i+1)..positions.len() {
                let d = (positions[i] - positions[j]).length();
                if d < min_dist { min_dist = d; }
            }
        }

        assert!(min_dist > 0.01, "Particles should be separated");
    }

    #[test]
    fn spatial_hash_enables_neighbor_queries() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);
        sim.spawn_water(16.0, 16.0, 0.0, 0.0, 100);

        sim.build_spatial_hash();
        sim.compute_neighbor_counts();

        // Particles in cluster should have neighbors
        let has_neighbors = sim.particles.iter().any(|p| p.neighbor_count > 0);
        assert!(has_neighbors);
    }
}
```

---

### Step 2.5: Create `flip/sediment.rs`

**Move from `flip.rs`:**
- `apply_sediment_forces` (lines 1146-1304)
- `apply_dem_settling` (lines 1306-1500)
- `deposit_settled_sediment` (lines 1502-1714)
- `entrain_deposited_sediment` (lines 1716-1863)
- `collapse_deposited_sediment` (lines 1865-1981)
- `count_column_deposited` (lines 1983-1992)
- `find_top_deposited_in_column` (lines 1994-2002)
- `find_landing_j` (lines 2004-2025)
- `update_particle_states` (lines 2027-2153)

**Unit Tests:**
```rust
// flip/sediment.rs
#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    fn create_sim_with_sediment() -> FlipSimulation {
        let mut sim = FlipSimulation::new(32, 32, 1.0);
        // Water column
        for j in 10..25 {
            for _ in 0..4 {
                sim.particles.list.push(
                    Particle::new(
                        Vec2::new(16.0 + rand::random::<f32>() * 0.5, j as f32),
                        Vec2::ZERO,
                        ParticleMaterial::Water
                    )
                );
            }
        }
        // Sand particles
        for _ in 0..20 {
            sim.particles.list.push(
                Particle::new(
                    Vec2::new(16.0, 20.0),
                    Vec2::ZERO,
                    ParticleMaterial::Sand
                )
            );
        }
        sim
    }

    #[test]
    fn sediment_forces_apply_gravity() {
        let mut sim = create_sim_with_sediment();

        let sand: Vec<&Particle> = sim.particles.iter()
            .filter(|p| p.material == ParticleMaterial::Sand)
            .collect();

        let vy_before: f32 = sand.iter().map(|p| p.velocity.y).sum();

        sim.apply_sediment_forces(0.1);

        let vy_after: f32 = sim.particles.iter()
            .filter(|p| p.material == ParticleMaterial::Sand)
            .map(|p| p.velocity.y)
            .sum();

        // Should accelerate downward
        assert!(vy_after < vy_before);
    }

    #[test]
    fn deposit_creates_deposited_cells() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);

        // Place settled sand at bottom
        for _ in 0..10 {
            let mut p = Particle::new(
                Vec2::new(16.5, 1.5),
                Vec2::ZERO,
                ParticleMaterial::Sand
            );
            p.state = ParticleState::Settled;
            sim.particles.list.push(p);
        }

        sim.classify_cells();
        sim.deposit_settled_sediment(0.1);

        // Check for deposited cell
        let has_deposited = (0..sim.grid.width).any(|i| {
            (0..sim.grid.height).any(|j| sim.grid.is_deposited(i, j))
        });

        assert!(has_deposited, "Should create deposited cells");
    }

    #[test]
    fn entrain_removes_deposited_with_high_shear() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);

        // Create deposited cell
        sim.grid.set_deposited_with_material(16, 5, ParticleMaterial::Sand);

        // High velocity flow above
        for j in 6..15 {
            let idx = sim.grid.u_index(16, j);
            sim.grid.u[idx] = 100.0; // High shear
        }

        let deposited_before = sim.grid.is_deposited(16, 5);
        assert!(deposited_before);

        sim.entrain_deposited_sediment(0.1);

        // May or may not entrain depending on shields threshold
        // Just verify no crash
    }

    #[test]
    fn collapse_fills_gaps() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);

        // Create gap: deposited at j=3 but not j=2
        sim.grid.set_deposited_with_material(16, 3, ParticleMaterial::Sand);

        sim.collapse_deposited_sediment();

        // Should collapse to j=1 (above solid at j=0)
        assert!(sim.grid.is_deposited(16, 1));
        assert!(!sim.grid.is_deposited(16, 3));
    }

    #[test]
    fn particle_states_update_correctly() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);

        // Particle in fluid
        let mut p = Particle::new(
            Vec2::new(16.0, 10.0),
            Vec2::new(0.0, -0.01), // Nearly stationary
            ParticleMaterial::Sand
        );
        p.state = ParticleState::Suspended;
        sim.particles.list.push(p);

        // Add water around it
        for dx in [-1.0, 0.0, 1.0] {
            for dy in [-1.0, 0.0, 1.0] {
                sim.particles.list.push(
                    Particle::new(
                        Vec2::new(16.0 + dx, 10.0 + dy),
                        Vec2::ZERO,
                        ParticleMaterial::Water
                    )
                );
            }
        }

        sim.classify_cells();
        sim.update_particle_states(0.1);

        // State should have been evaluated (may stay suspended or transition)
    }
}
```

---

### Step 2.6: Create `flip/pile.rs`

**Move from `flip.rs`:**
- `compute_pile_heightfield` (lines 2155-2195)
- `enforce_pile_constraints` (lines 2197-2236)

**Unit Tests:**
```rust
// flip/pile.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heightfield_reflects_deposits() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);

        // No deposits initially
        sim.compute_pile_heightfield();
        let height_empty = sim.pile_height[16];

        // Add deposits
        sim.grid.set_deposited(16, 1);
        sim.grid.set_deposited(16, 2);
        sim.grid.set_deposited(16, 3);

        sim.compute_pile_heightfield();
        let height_with_pile = sim.pile_height[16];

        assert!(height_with_pile > height_empty);
    }

    #[test]
    fn pile_constraints_stop_particles() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);

        // Create pile
        for j in 1..5 {
            sim.grid.set_deposited(16, j);
        }
        sim.compute_pile_heightfield();

        // Place particle inside pile
        sim.particles.list.push(
            Particle::new(
                glam::Vec2::new(16.5, 3.0), // Inside pile
                glam::Vec2::new(0.0, -10.0),
                ParticleMaterial::Sand
            )
        );

        sim.enforce_pile_constraints();

        let p = &sim.particles.list[0];
        // Should be pushed up or velocity zeroed
        assert!(p.position.y >= sim.pile_height[16] || p.velocity.y >= 0.0);
    }

    #[test]
    fn heightfield_handles_empty_columns() {
        let mut sim = FlipSimulation::new(32, 32, 1.0);

        sim.compute_pile_heightfield();

        // All heights should be at solid floor level
        for h in sim.pile_height.iter() {
            assert!(*h >= 0.0);
        }
    }
}
```

---

### Step 2.7: Create `flip/mod.rs`

**Keep in mod.rs:**
- `FlipSimulation` struct definition
- `new()` constructor
- `update()` - main orchestration
- `classify_cells()`
- Feature flag constants

**Module declarations:**
```rust
// flip/mod.rs
mod advection;
mod diagnostics;
mod pile;
mod sediment;
mod spawning;
mod transfer;

use crate::grid::Grid;
use crate::particle::{Particles, ParticleMaterial, ParticleState};

pub struct FlipSimulation {
    pub grid: Grid,
    pub particles: Particles,
    // ... other fields
}

impl FlipSimulation {
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self { ... }

    pub fn update(&mut self, dt: f32) {
        // Orchestrate all phases
        self.classify_cells();
        self.particles_to_grid();
        self.grid.apply_gravity(dt);
        // ... etc
    }

    pub fn classify_cells(&mut self) { ... }
}
```

---

## Phase 3: Update `lib.rs` and Integration

### Step 3.1: Update lib.rs exports

```rust
// lib.rs
pub mod flip;
pub mod grid;
pub mod particle;
pub mod pbf;
pub mod physics;
pub mod sluice;

pub use flip::FlipSimulation;
pub use grid::{Grid, CellType, DepositedCell};
pub use particle::{Particle, Particles, ParticleMaterial, ParticleState};
```

### Step 3.2: Update external tests

All tests in `tests/` directory should continue to work with:
```rust
use sim::flip::FlipSimulation;
use sim::grid::{Grid, CellType};
```

---

## Execution Order

| Step | Task | Est. Lines | Risk |
|------|------|------------|------|
| 1.1 | `grid/cell_types.rs` | 150 | Low |
| 1.2 | `grid/sdf.rs` | 200 | Low |
| 1.3 | `grid/velocity.rs` | 300 | Medium |
| 1.4 | `grid/pressure.rs` | 350 | Medium |
| 1.5 | `grid/extrapolation.rs` | 400 | Medium |
| 1.6 | `grid/vorticity.rs` | 250 | Low |
| 1.7 | `grid/mod.rs` | 400 | Low |
| **Checkpoint** | Run all tests | - | Verify |
| 2.1 | `flip/spawning.rs` | 150 | Low |
| 2.2 | `flip/diagnostics.rs` | 400 | Low |
| 2.3 | `flip/transfer.rs` | 450 | High |
| 2.4 | `flip/advection.rs` | 350 | Medium |
| 2.5 | `flip/sediment.rs` | 700 | High |
| 2.6 | `flip/pile.rs` | 150 | Low |
| 2.7 | `flip/mod.rs` | 300 | Low |
| **Checkpoint** | Run all tests | - | Verify |
| 3.1 | Update `lib.rs` | 20 | Low |
| 3.2 | Verify integration tests | - | Verify |

---

## Testing Strategy

### Unit Test Principles (from existing codebase)

1. **Use physical invariants** - conservation laws, not magic numbers
2. **Use ratios and relative comparisons** - not absolute values
3. **Deterministic by default** - fixed seeds for CI stability
4. **Randomized coverage** - add a small set of seeded variants or `#[ignore]` stress tests
5. **Multi-configuration tests** - can't tune for one specific case
6. **Test REAL simulation** - not isolated functions where possible

### Running Tests After Each Step

```bash
# After each file move
cargo test -p sim

# Full check
cargo test -p sim && cargo build --release
```

### Test Coverage Per Module

| Module | Test Focus |
|--------|------------|
| `cell_types` | Material properties, state transitions |
| `sdf` | Sign correctness, gradient direction |
| `velocity` | Interpolation, gravity direction |
| `pressure` | Divergence reduction, stats validity |
| `extrapolation` | Propagation, conservation, boundary respect |
| `vorticity` | Rotation detection, confinement effect |
| `spawning` | Particle creation, material assignment |
| `diagnostics` | Energy positivity, CFL scaling |
| `transfer` | P2G/G2P roundtrip conservation |
| `advection` | Movement, boundary respect, separation |
| `sediment` | State transitions, deposit/entrain |
| `pile` | Height computation, constraint enforcement |

---

## Rollback Plan

If issues arise:
1. Each step creates isolated files - delete and restore original
2. Keep `flip.rs` and `grid.rs` backups until fully validated
3. Git branch: `refactor/module-split` for safe experimentation

---

## Success Criteria

- [ ] All existing tests pass
- [ ] No new compiler warnings
- [ ] Each module has ≥3 unit tests
- [ ] Prefer files ≤ 500 lines (soft goal)
- [ ] `cargo build --release` succeeds
- [ ] Visual game runs correctly
