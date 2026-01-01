# FLIP/Grid Method Inventory for Parallel Refactoring

This document provides the explicit method-to-module mappings needed for parallel agent assignment.

---

## Grid Methods (grid.rs → grid/)

### grid/cell_types.rs

**Types to Move:**
- `CellType` (enum: Solid, Fluid, Air)
- `DepositedCell` (struct with material field)
- `MultigridLevel` (struct for multigrid hierarchy)

**Methods to Move:**

| Method | Signature | Notes |
|--------|-----------|-------|
| `DepositedCell::is_deposited` | `fn(&self) -> bool` | |
| `DepositedCell::effective_shields_critical` | `fn(&self) -> f32` | |
| `DepositedCell::effective_density` | `fn(&self) -> f32` | |
| `DepositedCell::color` | `fn(&self) -> [u8; 4]` | |
| `DepositedCell::clear` | `fn(&mut self)` | |
| `DepositedCell::get_material` | `fn(&self) -> Option<ParticleMaterial>` | |
| `MultigridLevel::new` | `fn(width: usize, height: usize) -> Self` | |
| `MultigridLevel::cell_index` | `fn(&self, i: usize, j: usize) -> usize` | |
| `Grid::pos_to_cell` | `fn(&self, pos: Vec2) -> (usize, usize)` | |
| `Grid::cell_index` | `fn(&self, i: usize, j: usize) -> usize` | |
| `Grid::u_index` | `fn(&self, i: usize, j: usize) -> usize` | |
| `Grid::v_index` | `fn(&self, i: usize, j: usize) -> usize` | |
| `Grid::set_solid` | `fn(&mut self, i: usize, j: usize)` | |
| `Grid::is_solid` | `fn(&self, i: usize, j: usize) -> bool` | |
| `Grid::set_deposited_with_material` | `fn(&mut self, i: usize, j: usize, material: ParticleMaterial)` | |
| `Grid::set_deposited` | `fn(&mut self, i: usize, j: usize)` | Legacy |
| `Grid::is_deposited` | `fn(&self, i: usize, j: usize) -> bool` | |
| `Grid::get_deposited` | `fn(&self, i: usize, j: usize) -> Option<&DepositedCell>` | |
| `Grid::clear_deposited` | `fn(&mut self, i: usize, j: usize)` | |
| `Grid::is_fluid` | `fn(&self, i: usize, j: usize) -> bool` | |

---

### grid/sdf.rs

| Method | Signature | Notes |
|--------|-----------|-------|
| `Grid::compute_sdf` | `fn(&mut self)` | Fast sweeping from solid cells |
| `Grid::sample_sdf` | `fn(&self, pos: Vec2) -> f32` | Bilinear interpolation |
| `Grid::sdf_gradient` | `fn(&self, pos: Vec2) -> Vec2` | Normalized gradient |
| `Grid::compute_bed_heights` | `fn(&mut self)` | Bed heightfield |
| `Grid::sample_bed_height` | `fn(&self, x: f32) -> f32` | Height interpolation |
| `Grid::normalized_height_above_bed` | `fn(&self, pos: Vec2, surface_height: f32) -> f32` | Normalized 0-1 |

**Dependencies:** Uses `self.solid`, `self.cell_type`, `self.inner_sdf`

---

### grid/velocity.rs

| Method | Signature | Notes |
|--------|-----------|-------|
| `Grid::sample_velocity` | `fn(&self, pos: Vec2) -> Vec2` | Bilinear interpolation |
| `Grid::sample_velocity_bspline` | `fn(&self, pos: Vec2) -> Vec2` | Quadratic B-spline |
| `Grid::sample_u` | `fn(&self, pos: Vec2) -> f32` | Private helper |
| `Grid::sample_v` | `fn(&self, pos: Vec2) -> f32` | Private helper |
| `Grid::sample_vorticity` | `fn(&self, pos: Vec2) -> f32` | Bilinear vorticity |
| `Grid::get_interp_weights` | `fn(&self, pos: Vec2) -> (usize, usize, [(i32, i32, f32); 4])` | |
| `Grid::clear_velocities` | `fn(&mut self)` | Zero field |
| `Grid::apply_gravity` | `fn(&mut self, dt: f32)` | Apply to V |
| `Grid::apply_viscosity` | `fn(&mut self, dt: f32, viscosity: f32)` | Explicit diffusion |
| `Grid::enforce_boundary_conditions` | `fn(&mut self)` | Zero at walls |
| `Grid::total_momentum` | `fn(&self) -> Vec2` | Grid momentum |
| `Grid::damp_surface_vertical` | `fn(&mut self)` | Damp V at surface |

**Dependencies:** Uses `self.u`, `self.v`, `self.cell_type`, `quadratic_bspline_1d()`

---

### grid/pressure.rs

| Method | Signature | Notes |
|--------|-----------|-------|
| `Grid::clear_pressure` | `fn(&mut self)` | Zero pressure/divergence |
| `Grid::compute_divergence` | `fn(&mut self)` | Velocity divergence |
| `Grid::total_divergence` | `fn(&self) -> f32` | Sum of |div| |
| `Grid::pressure_stats` | `fn(&self) -> (f32, f32, f32)` | Min, max, avg |
| `Grid::solve_pressure` | `fn(&mut self, iterations: usize)` | Red-Black GS |
| `Grid::update_pressure_cell` | `fn(&mut self, i: usize, j: usize, h_sq: f32)` | Private |
| `Grid::compute_max_residual` | `fn(&self, h_sq: f32) -> f32` | Convergence check |
| `Grid::apply_pressure_gradient` | `fn(&mut self, dt: f32)` | Subtract gradient |
| `Grid::solve_pressure_multigrid` | `fn(&mut self, num_cycles: usize)` | V-cycle solver |
| `Grid::mg_sync_level_zero` | `fn(&mut self)` | Private |
| `Grid::mg_copy_pressure_back` | `fn(&mut self)` | Private |
| `Grid::mg_restrict` | `fn(&mut self, fine: usize, coarse: usize)` | Private |
| `Grid::mg_prolongate` | `fn(&mut self, coarse: usize, fine: usize)` | Private |
| `Grid::mg_compute_residual` | `fn(&mut self, level: usize)` | Private |
| `Grid::mg_smooth` | `fn(&mut self, level: usize, iterations: usize)` | Private |
| `Grid::mg_clear_pressure` | `fn(&mut self, level: usize)` | Private |
| `Grid::mg_v_cycle` | `fn(&mut self, level: usize)` | Private |

**Dependencies:** Uses `self.pressure`, `self.divergence`, `self.cell_type`, `self.mg_levels`

---

### grid/extrapolation.rs

| Method | Signature | Notes |
|--------|-----------|-------|
| `Grid::extrapolate_velocities` | `fn(&mut self, max_layers: usize)` | Into air cells |
| `Grid::mark_fluid_faces_known` | `fn(&self, u_known: &mut [bool], v_known: &mut [bool])` | |
| `Grid::extrapolate_u_layer` | `fn(&mut self, u_known: &mut [bool])` | Single layer |
| `Grid::extrapolate_v_layer` | `fn(&mut self, v_known: &mut [bool])` | Single layer |
| `Grid::mark_fluid_faces_known_preallocated` | `fn(&mut self)` | Pre-allocated |
| `Grid::extrapolate_u_layer_preallocated` | `fn(&mut self)` | Pre-allocated |
| `Grid::extrapolate_v_layer_preallocated` | `fn(&mut self)` | Pre-allocated |

**Dependencies:** Uses `self.u_known`, `self.v_known`, `self.u_new_known`, `self.v_new_known`, `self.cell_type`

---

### grid/vorticity.rs

| Method | Signature | Notes |
|--------|-----------|-------|
| `Grid::compute_vorticity` | `fn(&mut self)` | Curl: dv/dx - du/dy |
| `Grid::compute_enstrophy` | `fn(&self) -> f32` | Total intensity |
| `Grid::total_absolute_vorticity` | `fn(&self) -> f32` | Sum of |omega| |
| `Grid::max_vorticity` | `fn(&self) -> f32` | Maximum magnitude |
| `Grid::apply_vorticity_confinement` | `fn(&mut self, dt: f32, strength: f32)` | Basic |
| `Grid::apply_vorticity_confinement_with_piles` | `fn(&mut self, dt: f32, strength: f32, pile_height: &[f32])` | With attenuation |

**Dependencies:** Uses `self.vorticity`, `self.u`, `self.v`, `self.cell_type`

---

### grid/interp.rs

**Free Functions (not methods):**

| Function | Signature | Notes |
|----------|-----------|-------|
| `quadratic_bspline_1d` | `fn(r: f32) -> f32` | 1D B-spline weight |
| `quadratic_bspline` | `fn(delta: Vec2) -> f32` | 2D B-spline weight |
| `apic_d_inverse` | `fn(cell_size: f32) -> f32` | APIC D matrix inverse |

**Re-export from grid/mod.rs**: These must be accessible as `crate::grid::quadratic_bspline` etc.

---

## FLIP Methods (flip.rs → flip/)

### flip/spawning.rs

| Method | Signature | Notes |
|--------|-----------|-------|
| `FlipSimulation::is_spawn_safe` | `fn(&self, x: f32, y: f32) -> bool` | Check position |
| `FlipSimulation::spawn_water` | `fn(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize)` | |
| `FlipSimulation::spawn_sand` | `fn(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize)` | |
| `FlipSimulation::spawn_magnetite` | `fn(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize)` | |
| `FlipSimulation::spawn_gold` | `fn(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize)` | |

**Dependencies:** Uses `self.grid.pos_to_cell()`, `self.grid.is_solid()`, `self.grid.sample_sdf()`, `self.particles.spawn_*()`, `rand::thread_rng()`

---

### flip/diagnostics.rs

| Method | Signature | Notes |
|--------|-----------|-------|
| `FlipSimulation::update_profiled` | `fn(&mut self, dt: f32) -> [f32; 7]` | Timing breakdown |
| `FlipSimulation::update_with_diagnostics` | `fn(&mut self, dt: f32) -> Vec<(&'static str, f32)>` | Momentum diagnostics |
| `FlipSimulation::run_isolated_flip_cycle` | `fn(&mut self, dt: f32) -> (f32, f32)` | Test cycle |
| `FlipSimulation::run_isolated_flip_cycle_with_extrapolation` | `fn(&mut self, dt: f32)` | With extrapolation |
| `FlipSimulation::compute_kinetic_energy` | `fn(&self) -> f32` | All particles |
| `FlipSimulation::compute_water_kinetic_energy` | `fn(&self) -> f32` | Water only |
| `FlipSimulation::compute_enstrophy` | `fn(&self) -> f32` | From vorticity |
| `FlipSimulation::update_and_compute_enstrophy` | `fn(&mut self) -> f32` | Classify + compute |
| `FlipSimulation::max_velocity` | `fn(&self) -> f32` | Maximum velocity |
| `FlipSimulation::compute_cfl` | `fn(&self, dt: f32) -> f32` | CFL number |
| `FlipSimulation::compute_grid_kinetic_energy` | `fn(&self) -> f32` | Grid-based KE |
| `FlipSimulation::get_u_weight_sum` | `fn(&self) -> f32` | P2G weight sum |
| `FlipSimulation::get_v_weight_sum` | `fn(&self) -> f32` | P2G weight sum |
| `FlipSimulation::initialize_taylor_green` | `fn(&mut self)` | Test setup |
| `FlipSimulation::initialize_solid_rotation` | `fn(&mut self, angular_velocity: f32)` | Test setup |

**Dependencies:** Calls most other FlipSimulation methods

---

### flip/transfer.rs (HIGH RISK)

| Method | Signature | Notes |
|--------|-----------|-------|
| `FlipSimulation::particles_to_grid` | `fn(&mut self)` | APIC P2G |
| `FlipSimulation::store_old_velocities` | `fn(&mut self)` | For FLIP delta |
| `FlipSimulation::grid_to_particles` | `fn(&mut self, dt: f32)` | APIC G2P |

**Dependencies:**
- Uses `quadratic_bspline_1d`, `quadratic_bspline`, `apic_d_inverse` from grid/interp
- Uses `self.u_sum`, `self.u_weight`, `self.v_sum`, `self.v_weight`
- Uses `self.sand_volume_u/v`, `self.water_volume_u/v` (two-way coupling)
- Uses `self.grid.u`, `self.grid.v`, `self.grid.u_index()`, `self.grid.v_index()`
- Uses `rayon::par_iter_mut()`

**CRITICAL:** Must use same kernel as `store_old_velocities` to avoid momentum loss.

---

### flip/advection.rs

| Method | Signature | Notes |
|--------|-----------|-------|
| `FlipSimulation::advect_particles` | `fn(&mut self, dt: f32)` | With SDF collision |
| `FlipSimulation::build_spatial_hash` | `fn(&mut self)` | Linked-cell list |
| `FlipSimulation::compute_neighbor_counts` | `fn(&mut self)` | For hindered settling |
| `FlipSimulation::compute_water_neighbor_counts` | `fn(&mut self)` | No-op (merged) |
| `FlipSimulation::push_particles_apart` | `fn(&mut self, iterations: usize)` | Soft separation |

**Dependencies:** Uses `self.grid.sample_sdf()`, `self.grid.sdf_gradient()`, `self.pile_height`, `self.cell_head`, `self.particle_next`, `rayon::par_iter_mut()`

---

### flip/sediment.rs (HIGH RISK)

| Method | Signature | Notes |
|--------|-----------|-------|
| `FlipSimulation::apply_sediment_forces` | `fn(&mut self, dt: f32)` | Drift-flux |
| `FlipSimulation::apply_dem_settling` | `fn(&mut self, dt: f32)` | DEM contacts |
| `FlipSimulation::deposit_settled_sediment` | `fn(&mut self, dt: f32)` | To solid cells |
| `FlipSimulation::entrain_deposited_sediment` | `fn(&mut self, dt: f32)` | Erosion |
| `FlipSimulation::collapse_deposited_sediment` | `fn(&mut self)` | Gravity collapse |
| `FlipSimulation::count_column_deposited` | `fn(&self, i: usize) -> usize` | |
| `FlipSimulation::find_top_deposited_in_column` | `fn(&self, i: usize) -> Option<usize>` | |
| `FlipSimulation::find_landing_j` | `fn(&self, i: usize) -> usize` | |

**Dependencies:** Uses `hindered_settling_factor`, `neighbor_count_to_concentration` from particle, `self.neighbor_counts`, `self.grid.sample_sdf()`, `self.grid.is_deposited()`, `self.deposited_mass_*`, `rayon::par_iter_mut()`, `rand::thread_rng()`

---

### flip/pile.rs

| Method | Signature | Notes |
|--------|-----------|-------|
| `FlipSimulation::update_particle_states` | `fn(&mut self, dt: f32)` | Suspended/Bedload |
| `FlipSimulation::compute_pile_heightfield` | `fn(&mut self)` | From bedload |
| `FlipSimulation::enforce_pile_constraints` | `fn(&mut self)` | Push above pile |

**Dependencies:** Uses `self.pile_height`, `self.grid.sample_sdf()`, `self.grid.sdf_gradient()`, `self.grid.sample_velocity()`, `rayon::par_iter_mut()`

---

### flip/pressure.rs (FlipSimulation side)

| Method | Signature | Notes |
|--------|-----------|-------|
| `FlipSimulation::apply_pressure_gradient_two_way` | `fn(&mut self, dt: f32)` | Mixture density |
| `FlipSimulation::apply_porosity_drag` | `fn(&mut self, dt: f32)` | Dense particle drag |

**Dependencies:** Uses `self.sand_volume_u/v`, `self.water_volume_u/v`, `self.grid.pressure`, `self.grid.cell_type`

---

### flip/mod.rs (Keep in parent)

| Method | Signature | Notes |
|--------|-----------|-------|
| `FlipSimulation::new` | `fn(width: usize, height: usize, cell_size: f32) -> Self` | Constructor |
| `FlipSimulation::update` | `fn(&mut self, dt: f32)` | Main step |
| `FlipSimulation::classify_cells` | `fn(&mut self)` | Solid/Fluid/Air |

**Note:** `update()` orchestrates all other methods and should stay in mod.rs.

---

## Shared Types Across Modules

### From particle.rs (unchanged)
- `ParticleState` - Suspended, Bedload
- `ParticleMaterial` - Water, Mud, Sand, Magnetite, Gold
- `Particle` - Full particle struct
- `Particles` - Collection
- `hindered_settling_factor(concentration: f32) -> f32`
- `neighbor_count_to_concentration(count: usize, rest: f32) -> f32`

### From physics.rs (unchanged)
- `GRAVITY: f32 = 350.0`
- `WATER_DENSITY: f32 = 1.0`
- `KINEMATIC_VISCOSITY: f32 = 1.3`

---

## Cross-Module Dependency Matrix

This matrix shows which FLIP modules call which Grid modules:

| FLIP Module | Grid Modules Called |
|-------------|---------------------|
| transfer | interp, cell_types, velocity |
| advection | sdf, cell_types |
| sediment | sdf, cell_types, velocity |
| pile | sdf, velocity |
| pressure | cell_types, pressure |
| spawning | sdf, cell_types |
| diagnostics | vorticity (+ all others via update) |

---

## Agent Assignment Checklist Template

When an agent claims a module, they should fill out:

```markdown
## Agent Claim: [Module Name]

**Agent ID:** [e.g., G1, F3]
**Branch:** refactor/[module]-[agent]
**Status:** [CLAIMED | WIP | READY_FOR_REVIEW | MERGED]

### Methods Cut List
- [ ] `method_name_1`
- [ ] `method_name_2`
- ...

### Files Modified
- [ ] `grid.rs` or `flip.rs` (cut methods)
- [ ] `grid/[module].rs` or `flip/[module].rs` (paste methods)
- [ ] `grid/mod.rs` or `flip/mod.rs` (add module declaration)

### Dependencies on Other Agents
- Requires: [list agent IDs this depends on]
- Blocks: [list agent IDs that depend on this]

### Tests Added
- [ ] `test_[something]`
- [ ] `test_[something_else]`

### Pre-Merge Checklist
- [ ] `cargo build -p sim` passes
- [ ] `cargo test -p sim` passes
- [ ] `cargo clippy -p sim -- -D warnings` passes
- [ ] Rebased on latest main
```
