# FLIP Simulation Collision Systems

## Overview

This document maps all collision/solid-boundary handling systems in `flip.rs` and `grid.rs`. Understanding these is critical before making performance changes.

## Solid Data Structures

### 1. `grid.solid: Vec<bool>` (grid.rs)
- **Purpose**: Boolean mask of solid cells
- **Set by**: `sluice.rs` via `grid.set_solid(i, j)`
- **Accessed via**: `grid.is_solid(i, j)` - O(1) array lookup
- **Size**: `width * height` cells

### 2. `grid.sdf: Vec<f32>` (grid.rs)
- **Purpose**: Signed distance field to nearest solid (negative inside, positive outside)
- **Computed by**: `grid.compute_sdf()` - Fast sweeping O(n) where n = grid cells
- **Accessed via**:
  - `grid.sample_sdf(pos)` - O(1) bilinear interpolation
  - `grid.sdf_gradient(pos)` - O(1) via 4 sample_sdf calls
- **When recomputed**: After terrain changes (called in `sluice.rs`)

### 3. `grid.cell_type: Vec<CellType>` (grid.rs)
- **Purpose**: Per-frame classification (Solid/Fluid/Air) for pressure solver
- **Set by**: `classify_cells()` each frame
- **Used by**: Pressure solver, divergence computation

---

## Collision Functions (What Uses What)

### A. Per-Frame Functions (called in update())

| Function | Solid Checks | Scales With | Notes |
|----------|--------------|-------------|-------|
| `classify_cells()` | `is_solid()` on all cells | Grid size | O(w*h) - independent of riffle count |
| `enforce_boundary_conditions()` | `is_solid()` on all faces | Grid size | O(w*h) - independent of riffle count |
| `update_pressure_cell()` | Uses `cell_type[]` | Grid size | No is_solid calls - uses precomputed |
| `advect_particles()` | **SDF** `sample_sdf()` | Particle count | O(1) per particle - **riffle-independent** |

### B. Every-2-Frames Functions

| Function | Solid Checks | Scales With | Notes |
|----------|--------------|-------------|-------|
| `apply_near_pressure()` | None | Particle^2 / h^2 | Neighbor search via spatial hash |

### C. Dead Code (NOT CALLED - line 131 says removed)

| Function | Would Use | Notes |
|----------|-----------|-------|
| `separate_particles()` | `is_solid()` on corrections | Not called - Clavet handles separation |
| `resolve_solid_penetration()` | `is_solid()` on neighbors | Only called by separate_particles |

---

## The SDF Implementation

### How It Works
```
compute_sdf():
1. Initialize: solid cells = 0, non-solid = MAX
2. Fast sweeping: 4 passes (forward/backward) × 4 sweeps
3. Propagates minimum distance from solid boundaries
```

### Usage in advect_particles():
```rust
// Sample distance to nearest solid
let sdf_dist = grid.sample_sdf(particle.position);

if sdf_dist < cell_size * 0.5 {
    // Get gradient (direction away from solid)
    let grad = grid.sdf_gradient(particle.position);
    // Push out to safe distance
    particle.position += grad * (cell_size * 0.5 - sdf_dist);
    // Remove velocity into solid
    if velocity.dot(grad) < 0.0 {
        velocity -= grad * velocity.dot(grad);
    }
}
```

---

## Performance Characteristics

### What scales with RIFFLE COUNT:
- **Nothing in collision detection** - SDF is precomputed once
- More riffles = more solid cells = different *fluid behavior* (more turbulence, bouncing)

### What scales with PARTICLE COUNT:
- `advect_particles()` - O(n) particles × O(1) SDF lookups
- `apply_near_pressure()` - O(n) × O(neighbors) via spatial hash
- `particles_to_grid()` - O(n) particles
- `grid_to_particles()` - O(n) particles

### What scales with GRID SIZE:
- `classify_cells()` - O(w*h)
- `enforce_boundary_conditions()` - O(w*h)
- `solve_pressure()` - O(iterations * w * h)
- `compute_sdf()` - O(w*h) but only on terrain change, not per frame

---

## Identified Conflicts / Issues

### 1. Duplicate Collision Systems
- `advect_particles()` now uses SDF
- `resolve_solid_penetration()` still uses is_solid() - BUT it's dead code
- `separate_particles()` still uses is_solid() - BUT it's dead code
- **Status**: No conflict - dead code not called

### 2. enforce_boundary_conditions Still Uses is_solid()
- Iterates all u-faces and v-faces checking is_solid
- This is O(grid size) regardless of riffle count
- **Not a performance issue** - is_solid is O(1) lookup

### 3. classify_cells Still Uses is_solid()
- Iterates all cells checking is_solid
- **Could be optimized**: Store list of solid cell indices, iterate that instead
- **Impact**: Minimal - O(w*h) with O(1) lookups

---

## Why Riffle Count Still Affects Performance

If collision detection is riffle-independent (SDF), why does spacing=10 still slow down?

### Hypothesis 1: Fluid Dynamics Behavior
More riffles create:
- More turbulence (particles bouncing between riffles)
- Higher velocities in confined spaces
- More pressure iterations needed to converge

### Hypothesis 2: Pressure Solver Convergence
- More solid boundaries = more complex pressure field
- 10 iterations may not be enough
- Divergence may be higher, requiring more computation

### Hypothesis 3: Near-Pressure (Clavet) Work
- More turbulence = more particle clustering
- More neighbors = more work in apply_near_pressure()

---

## Recommendations Before Making Changes

1. **Profile first**: Use `cargo flamegraph` to identify actual bottleneck
2. **Measure divergence**: Is it higher with more riffles?
3. **Check near-pressure**: Is neighbor count higher?
4. **Don't remove dead code yet**: It may be useful for debugging
