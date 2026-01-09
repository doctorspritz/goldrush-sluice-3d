<!-- TODO: Review -->

# Plan: Implement Velocity Extrapolation

**Date:** 2025-12-28
**Status:** PENDING REVIEW
**Category:** feature, physics, momentum-conservation

## Problem Statement

Water flows like honey due to momentum loss at fluid boundaries. Particles near air/solid cells sample undefined or zeroed velocities, causing phantom FLIP deltas.

## Solution: Layered Wavefront Velocity Extrapolation

### Algorithm Choice

**Layered Wavefront** (not full Fast Marching) because:
- Simpler to implement
- Sufficient for FLIP (1-2 cell extrapolation needed)
- Used by production solvers (Houdini)
- Easy to debug and understand

Can upgrade to FMM later if needed.

### Algorithm

```
extrapolate_velocities(max_layers):
    mark all FLUID cells as KNOWN

    for layer in 0..max_layers:
        for each cell:
            if cell is AIR and has any KNOWN neighbor:
                cell.velocity = weighted_average(known_neighbors)
                mark cell as KNOWN_THIS_LAYER

        promote KNOWN_THIS_LAYER to KNOWN
```

For MAC grid (staggered velocities):
- Extrapolate U and V components separately
- U faces: check left/right fluid cells
- V faces: check above/below fluid cells

---

## Implementation Steps

### Step 1: Add `extrapolate_velocities()` to Grid

**File:** `crates/sim/src/grid.rs`

```rust
/// Extrapolate velocities from fluid cells into non-fluid cells
/// Uses layered wavefront: each layer copies from known neighbors
pub fn extrapolate_velocities(&mut self, max_layers: usize) {
    // Track which cells have valid velocities
    let mut u_known: Vec<bool> = vec![false; self.u.len()];
    let mut v_known: Vec<bool> = vec![false; self.v.len()];

    // Initialize: fluid cell faces are known
    self.mark_fluid_faces_known(&mut u_known, &mut v_known);

    // Propagate layer by layer
    for _ in 0..max_layers {
        self.extrapolate_u_layer(&mut u_known);
        self.extrapolate_v_layer(&mut v_known);
    }
}
```

#### 1a. mark_fluid_faces_known (DETAILED)

```rust
fn mark_fluid_faces_known(&self, u_known: &mut Vec<bool>, v_known: &mut Vec<bool>) {
    // Mark U faces (between cells i-1 and i)
    // A U face is "known" if EITHER adjacent cell is fluid
    // BUT NOT if adjacent to solid (don't extrapolate from solid boundaries)
    for j in 0..self.height {
        for i in 0..=self.width {
            let u_idx = self.u_index(i, j);

            // Check cells on either side
            let left_fluid = i > 0 && self.cell_type[self.cell_index(i-1, j)] == CellType::Fluid;
            let right_fluid = i < self.width && self.cell_type[self.cell_index(i, j)] == CellType::Fluid;

            // Don't mark if touching solid (solid boundaries handled separately)
            let left_solid = i == 0 || self.cell_type[self.cell_index(i-1, j)] == CellType::Solid;
            let right_solid = i == self.width || self.cell_type[self.cell_index(i, j)] == CellType::Solid;

            // Known if has fluid neighbor and not blocked by solid on both sides
            if (left_fluid || right_fluid) && !(left_solid && right_solid) {
                u_known[u_idx] = true;
            }
        }
    }

    // Mark V faces (between cells j-1 and j) - same logic
    for j in 0..=self.height {
        for i in 0..self.width {
            let v_idx = self.v_index(i, j);

            let bottom_fluid = j > 0 && self.cell_type[self.cell_index(i, j-1)] == CellType::Fluid;
            let top_fluid = j < self.height && self.cell_type[self.cell_index(i, j)] == CellType::Fluid;

            let bottom_solid = j == 0 || self.cell_type[self.cell_index(i, j-1)] == CellType::Solid;
            let top_solid = j == self.height || self.cell_type[self.cell_index(i, j)] == CellType::Solid;

            if (bottom_fluid || top_fluid) && !(bottom_solid && top_solid) {
                v_known[v_idx] = true;
            }
        }
    }
}
```

#### 1b. extrapolate_u_layer (DETAILED)

```rust
fn extrapolate_u_layer(&mut self, u_known: &mut Vec<bool>) {
    // Two-buffer pattern: compute new values without reading them in same pass
    let mut new_known = vec![false; u_known.len()];
    let mut new_values = vec![0.0f32; self.u.len()];

    for j in 0..self.height {
        for i in 0..=self.width {
            let u_idx = self.u_index(i, j);

            if u_known[u_idx] {
                continue; // Already known, skip
            }

            // Average of known cardinal neighbors (staggered U grid)
            let mut sum = 0.0;
            let mut count = 0;

            // Left neighbor (i-1, j)
            if i > 0 {
                let idx = self.u_index(i-1, j);
                if u_known[idx] {
                    sum += self.u[idx];
                    count += 1;
                }
            }
            // Right neighbor (i+1, j)
            if i < self.width {
                let idx = self.u_index(i+1, j);
                if u_known[idx] {
                    sum += self.u[idx];
                    count += 1;
                }
            }
            // Down neighbor (i, j-1)
            if j > 0 {
                let idx = self.u_index(i, j-1);
                if u_known[idx] {
                    sum += self.u[idx];
                    count += 1;
                }
            }
            // Up neighbor (i, j+1)
            if j + 1 < self.height {
                let idx = self.u_index(i, j+1);
                if u_known[idx] {
                    sum += self.u[idx];
                    count += 1;
                }
            }

            if count > 0 {
                new_values[u_idx] = sum / count as f32;
                new_known[u_idx] = true;
            }
            // If count == 0, leave unchanged (next layer may reach it)
        }
    }

    // Apply new values and merge known flags
    for i in 0..u_known.len() {
        if new_known[i] {
            self.u[i] = new_values[i];
            u_known[i] = true;
        }
    }
}
```

#### 1c. extrapolate_v_layer (DETAILED)

```rust
fn extrapolate_v_layer(&mut self, v_known: &mut Vec<bool>) {
    let mut new_known = vec![false; v_known.len()];
    let mut new_values = vec![0.0f32; self.v.len()];

    for j in 0..=self.height {
        for i in 0..self.width {
            let v_idx = self.v_index(i, j);

            if v_known[v_idx] {
                continue;
            }

            let mut sum = 0.0;
            let mut count = 0;

            // Left neighbor (i-1, j)
            if i > 0 {
                let idx = self.v_index(i-1, j);
                if v_known[idx] {
                    sum += self.v[idx];
                    count += 1;
                }
            }
            // Right neighbor (i+1, j)
            if i + 1 < self.width {
                let idx = self.v_index(i+1, j);
                if v_known[idx] {
                    sum += self.v[idx];
                    count += 1;
                }
            }
            // Down neighbor (i, j-1)
            if j > 0 {
                let idx = self.v_index(i, j-1);
                if v_known[idx] {
                    sum += self.v[idx];
                    count += 1;
                }
            }
            // Up neighbor (i, j+1)
            if j <= self.height {
                let idx = self.v_index(i, j+1);
                if v_known[idx] {
                    sum += self.v[idx];
                    count += 1;
                }
            }

            if count > 0 {
                new_values[v_idx] = sum / count as f32;
                new_known[v_idx] = true;
            }
        }
    }

    for i in 0..v_known.len() {
        if new_known[i] {
            self.v[i] = new_values[i];
            v_known[i] = true;
        }
    }
}
```

### Step 2: Update Simulation Loop

**File:** `crates/sim/src/flip.rs`

**Before:**
```rust
// 2. Transfer particle velocities to grid (P2G)
self.particles_to_grid();

// 3. Store old grid velocities for FLIP blending
self.store_old_velocities();
```

**After:**
```rust
// 2. Transfer particle velocities to grid (P2G)
self.particles_to_grid();

// 2b. Extrapolate velocities into air cells (1 layer)
// This ensures store_old_velocities samples valid values everywhere
self.grid.extrapolate_velocities(1);

// 3. Store old grid velocities for FLIP blending
self.store_old_velocities();
```

**Also after pressure solve:**
```rust
self.grid.apply_pressure_gradient(dt);

// 5c. Re-extrapolate after pressure changed velocities
self.grid.extrapolate_velocities(1);

// 6. Transfer grid velocities back to particles
self.grid_to_particles(dt);
```

### Step 3: Update `run_isolated_flip_cycle()` Diagnostic

**File:** `crates/sim/src/flip.rs`

Add extrapolation to match real simulation:
```rust
pub fn run_isolated_flip_cycle_with_extrapolation(&mut self, dt: f32) -> (f32, f32) {
    let momentum_before = self.total_particle_momentum();

    self.classify_cells();
    self.particles_to_grid();
    self.grid.extrapolate_velocities(1);  // NEW
    self.store_old_velocities();
    // NO forces
    self.grid_to_particles(dt);

    let momentum_after = self.total_particle_momentum();
    (momentum_before, momentum_after)
}
```

---

## What This Does NOT Touch

- FLIP_RATIO (0.97) - unchanged
- Pressure solver - unchanged
- Boundary enforcement - unchanged (still needed after forces)
- Sediment physics - unchanged
- Any damping constants - unchanged
- Collision handling - unchanged

---

## Test Plan

All tests in `crates/sim/tests/velocity_extrapolation_tests.rs`:

1. **flip_cycle_conserves_momentum_with_extrapolation** - ratio > 0.99, 10 random seeds
2. **no_phantom_delta_at_air_boundary** - delta < 1% of velocity, 5 random positions
3. **extrapolation_does_not_change_grid_momentum** - relative change < 1%
4. **extrapolated_velocity_is_neighbor_average** - value within neighbor range
5. **multi_layer_extrapolation_propagates** - layer 3 > layer 1
6. **momentum_stable_over_sixty_frames** - 80% retained after 60 frames (catches 2% per-frame loss)
7. **full_step_with_extrapolation_conserves_momentum** - X momentum ratio 0.95-1.05
8. **extrapolation_respects_solid_boundaries** - no velocity leak into solids
9. **extrapolation_performance_within_budget** - <1ms per call (60fps requirement)
10. **conservation_across_grid_sizes** - works for 16, 32, 64 grid sizes

**Anti-cheat properties:**
- Random seeds and positions (can't tune for specific case)
- Ratio-based thresholds (can't game absolute values)
- Multiple configurations (can't hack one scenario)
- Physical invariants (conservation laws MUST hold)
- Performance test (can't fake wall-clock time)

**DO NOT modify test thresholds to make implementation pass. Fix the implementation.**

---

## Performance Requirements

**Target: 60fps minimum (16.6ms per frame)**

Extrapolation budget: <1ms per call (called twice per frame = <2ms total)

**Complexity analysis:**
- Grid traversal: O(width × height) per layer
- With 64×48 grid, 2 layers: ~6000 cells × 2 layers × 2 components = ~24000 ops
- Each op: 4 neighbor checks + 1 average = ~5 operations = ~120k total
- At modern CPU speeds (~3GHz), this is <100 microseconds

**If performance is an issue:**
1. Reduce to 1 layer (often sufficient)
2. Parallelize with rayon (each row independent)
3. Skip cells far from fluid (add distance check)

**DO NOT sacrifice correctness for performance. Fix algorithm first, optimize later.**

---

## Required Methods (Tests Will Fail Until These Exist)

The tests require these methods to be implemented:

### Grid (grid.rs)
```rust
impl MacGrid {
    /// Extrapolate velocities from fluid cells into non-fluid cells
    pub fn extrapolate_velocities(&mut self, max_layers: usize);

    /// Total momentum vector of the grid (for conservation tests)
    pub fn total_momentum(&self) -> Vec2;
}
```

### FlipSimulation (flip.rs)
```rust
impl FlipSimulation {
    /// Run isolated FLIP cycle WITH extrapolation for testing
    /// P2G → extrapolate → store_old → G2P (NO forces)
    pub fn run_isolated_flip_cycle_with_extrapolation(&mut self, dt: f32);

    /// Total momentum of all water particles
    pub fn total_particle_momentum(&self) -> f32;
}
```

**The tests will fail to compile until these methods exist. That's correct TDD.**

---

## Original Test Plan

From `velocity-extrapolation-test-design.md`:

1. **flip_cycle_conserves_momentum_with_extrapolation** - ratio > 0.99
3. **tangential_velocity_preserved_at_walls** - ratio > 0.95
4. **momentum_stable_over_time** - 90% retained after 60 frames
5. **extrapolated_velocity_matches_neighbor_average** - algorithm correctness
6. **water_accelerates_downhill_in_sluice** - real game behavior

**Write tests FIRST, then implement.**

---

## Order of Work

1. Write test file `crates/sim/tests/velocity_extrapolation_tests.rs`
2. Run tests (they should FAIL - no implementation yet)
3. Implement `extrapolate_velocities()` in grid.rs
4. Update simulation loop in flip.rs
5. Run tests (they should PASS)
6. Run real game and verify water flows naturally

---

## Risk Assessment

**Medium risk:**
- Changes simulation loop order (adding steps)
- Could affect performance (extra grid traversal)
- Need to handle edge cases (corners, single-cell fluid regions)

**Mitigations:**
- Tests catch regressions
- Performance: extrapolation is O(N) and parallelizable
- Edge cases: use max of known neighbors, fallback to zero

---

## Performance Notes

- Extrapolation is O(width × height) per layer
- With 1 layer and 64×64 grid: 4096 cells × 2 passes = 8192 ops
- This is negligible compared to pressure solve (15 iterations × 4096 × 2)
- Can be parallelized with rayon if needed

---

## References

- [Velocity Extrapolation Research](../research/velocity-extrapolation-research.md)
- [Test Design Document](../research/velocity-extrapolation-test-design.md)
- [WebGL-PIC-FLIP-Fluid](https://github.com/austinEng/WebGL-PIC-FLIP-Fluid)
- [Houdini FLIP Solver](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html)
