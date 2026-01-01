# perf: Improve PIC/FLIP Pressure Solver, Collision, and Performance

## Overview

Upgrade the fluid simulation's core numerical components based on code review feedback:
1. **Pressure Solver**: Switch from Jacobi to Red-Black Gauss-Seidel for 2x faster convergence
2. **Collision Detection**: Fix particle tunneling through thin walls at high velocities
3. **Spatial Hash**: Replace HashMap with linked-cell list (zero allocation per frame)
4. **Sub-stepping**: Multiple physics steps per frame for stability
5. **Softer Particle Separation**: Force-based instead of hard position correction

## Problem Statement

### Current Pressure Solver Issues
- **Algorithm**: Jacobi iteration requires 80 iterations per frame (`grid.rs:48`)
- **Convergence**: Jacobi is slow to converge - each iteration only uses "stale" neighbor values
- **Memory**: Allocates temporary `new_pressure` buffer every frame (`grid.rs:243`)

### Current Collision Detection Issues
- **Tunneling Risk**: Particles moving faster than `cell_size` per frame can skip through walls
- **Discrete Detection**: Only checks final position, not the path traveled (`flip.rs:206`)
- **No Velocity Limiting**: Particles can have arbitrarily high velocities

### Current Performance Issues (8 FPS @ 2500 particles)
- **Memory Allocation**: `separate_particles()` creates HashMap every frame - thousands of allocations/s
- **Hard Position Correction**: Causes jittery, clunky flow - particles "pop" instead of flowing
- **Single Large Timestep**: Contributes to instability and visual jitter

## Proposed Solution

### Phase 1: Red-Black Gauss-Seidel Pressure Solver

Replace Jacobi with Red-Black Gauss-Seidel for ~2x convergence improvement:

```rust
// grid.rs - Replace solve_pressure()
pub fn solve_pressure_rbgs(&mut self, iterations: usize) {
    for _ in 0..iterations {
        // Red pass (i+j even) - uses latest neighbor values
        for j in 1..self.height - 1 {
            for i in 1..self.width - 1 {
                if (i + j) % 2 == 0 {
                    self.update_pressure_cell(i, j);
                }
            }
        }
        // Black pass (i+j odd) - uses updated red values
        for j in 1..self.height - 1 {
            for i in 1..self.width - 1 {
                if (i + j) % 2 != 0 {
                    self.update_pressure_cell(i, j);
                }
            }
        }
    }
}

fn update_pressure_cell(&mut self, i: usize, j: usize) {
    let idx = self.cell_index(i, j);

    if self.cell_type[idx] != CellType::Fluid {
        self.pressure[idx] = 0.0;
        return;
    }

    // Get neighbor pressures (in-place - no temp buffer needed!)
    let p_left = self.get_pressure_or_reflect(i.wrapping_sub(1), j, idx);
    let p_right = self.get_pressure_or_reflect(i + 1, j, idx);
    let p_bottom = self.get_pressure_or_reflect(i, j.wrapping_sub(1), idx);
    let p_top = self.get_pressure_or_reflect(i, j + 1, idx);

    self.pressure[idx] = (p_left + p_right + p_bottom + p_top - self.divergence[idx]) * 0.25;
}
```

**Benefits**:
- ~2x faster convergence (can reduce iterations from 80 to 40)
- No temporary buffer allocation (in-place update)
- Same code complexity

### Phase 2: Collision Detection Improvements

#### 2a. Velocity Clamping (CFL Condition)

Enforce maximum velocity to prevent tunneling:

```rust
// flip.rs - Add after grid_to_particles()
fn clamp_velocities(&mut self) {
    let max_velocity = self.grid.cell_size * 0.9 / (1.0 / 60.0); // 90% of cell per frame

    for particle in self.particles.iter_mut() {
        let speed = particle.velocity.length();
        if speed > max_velocity {
            particle.velocity = particle.velocity.normalize() * max_velocity;
        }
    }
}
```

#### 2b. Sub-stepped Advection

For particles that would exceed the CFL limit, use sub-stepping:

```rust
// flip.rs - Replace advect_particles()
fn advect_particles(&mut self, dt: f32) {
    let cell_size = self.grid.cell_size;
    let max_dist_per_step = cell_size * 0.5; // Max half-cell per substep

    for particle in self.particles.iter_mut() {
        let total_dist = (particle.velocity * dt).length();
        let substeps = ((total_dist / max_dist_per_step).ceil() as usize).max(1);
        let sub_dt = dt / substeps as f32;

        for _ in 0..substeps {
            particle.position += particle.velocity * sub_dt;
            resolve_solid_collision(particle, &self.grid, cell_size, width, height);
        }
    }
}
```

#### 2c. Ray-Based Collision (Optional Enhancement)

For maximum robustness, trace path through grid:

```rust
// flip.rs - Enhanced collision for fast particles
fn advect_with_raycast(particle: &mut Particle, grid: &Grid, dt: f32) {
    let start = particle.position;
    let end = start + particle.velocity * dt;

    // Walk through cells along path using DDA algorithm
    let mut t = 0.0;
    while t < 1.0 {
        let pos = start.lerp(end, t);
        let (i, j) = grid.pos_to_cell(pos);

        if grid.is_solid(i, j) {
            // Hit solid - stop just before
            particle.position = start.lerp(end, (t - 0.01).max(0.0));
            // Zero velocity component into wall
            // ... (determine wall normal and reflect)
            return;
        }

        // Step to next cell boundary
        t += grid.cell_size / (end - start).length();
    }

    particle.position = end;
}
```

### Phase 3: Linked-Cell List (Zero-Allocation Spatial Hash)

Replace HashMap with pre-allocated arrays for O(0) allocations per frame:

```rust
// flip.rs - Add fields to FlipSimulation
pub struct FlipSimulation {
    pub grid: Grid,
    pub particles: Particles,
    // Pre-allocated spatial hash
    cell_head: Vec<i32>,      // Index of first particle in each cell (-1 = empty)
    particle_next: Vec<i32>,  // Index of next particle in same cell (-1 = end)
}

impl FlipSimulation {
    fn build_spatial_hash(&mut self) {
        // Clear cell heads (no allocation!)
        self.cell_head.fill(-1);

        // Resize particle_next if needed
        if self.particle_next.len() < self.particles.len() {
            self.particle_next.resize(self.particles.len(), -1);
        }

        // Build linked lists
        for (idx, particle) in self.particles.list.iter().enumerate() {
            let gi = (particle.position.x / self.grid.cell_size) as usize;
            let gj = (particle.position.y / self.grid.cell_size) as usize;
            let cell_idx = gj * self.grid.width + gi;

            // Insert at head of list
            self.particle_next[idx] = self.cell_head[cell_idx];
            self.cell_head[cell_idx] = idx as i32;
        }
    }

    fn for_each_neighbor<F>(&self, particle_idx: usize, mut f: F)
    where F: FnMut(usize)
    {
        let pos = self.particles.list[particle_idx].position;
        let gi = (pos.x / self.grid.cell_size) as i32;
        let gj = (pos.y / self.grid.cell_size) as i32;

        // Check 3x3 neighborhood
        for dj in -1..=1 {
            for di in -1..=1 {
                let ni = gi + di;
                let nj = gj + dj;
                if ni < 0 || nj < 0 || ni >= self.grid.width as i32 || nj >= self.grid.height as i32 {
                    continue;
                }
                let cell_idx = (nj as usize) * self.grid.width + (ni as usize);

                // Walk linked list
                let mut idx = self.cell_head[cell_idx];
                while idx >= 0 {
                    if idx as usize != particle_idx {
                        f(idx as usize);
                    }
                    idx = self.particle_next[idx as usize];
                }
            }
        }
    }
}
```

**Benefits**:
- Zero heap allocations per frame
- Cache-friendly iteration
- Same O(n) complexity

### Phase 4: Sub-Stepping for Stability

Run multiple smaller physics steps per frame:

```rust
// flip.rs - Modify update()
pub fn update(&mut self, dt: f32) {
    const SUBSTEPS: usize = 4;
    let sub_dt = dt / SUBSTEPS as f32;

    for _ in 0..SUBSTEPS {
        self.update_substep(sub_dt);
    }
}

fn update_substep(&mut self, dt: f32) {
    self.classify_cells();
    self.particles_to_grid();
    self.store_old_velocities();
    self.grid.apply_gravity(dt);
    self.grid.compute_divergence();
    self.grid.solve_pressure_rbgs(20); // Fewer iterations per substep
    self.grid.apply_pressure_gradient(dt);
    self.grid.apply_vorticity_confinement(dt, 5.0);
    self.grid_to_particles();
    self.apply_settling(dt);
    self.advect_particles(dt);
    self.separate_particles_soft(dt);
}
```

**Benefits**:
- Smoother motion (4x smaller position changes)
- Prevents tunneling naturally
- Allows lower iteration counts per substep

### Phase 5: Soft Particle Separation

Replace hard position correction with force-based separation:

```rust
// flip.rs - Replace separate_particles()
fn separate_particles_soft(&mut self, dt: f32) {
    self.build_spatial_hash();

    let cell_size = self.grid.cell_size;
    let min_dist = cell_size * 0.8;
    let stiffness = 500.0; // Separation force strength

    // Collect velocity adjustments
    let mut velocity_deltas: Vec<Vec2> = vec![Vec2::ZERO; self.particles.len()];

    for idx in 0..self.particles.len() {
        let pos = self.particles.list[idx].position;

        self.for_each_neighbor(idx, |other_idx| {
            let other_pos = self.particles.list[other_idx].position;
            let diff = pos - other_pos;
            let dist = diff.length();

            if dist < min_dist && dist > 0.001 {
                let overlap = min_dist - dist;
                let dir = diff / dist;
                // Apply as velocity impulse, not position correction
                let force = dir * overlap * stiffness;
                velocity_deltas[idx] += force * dt;
            }
        });
    }

    // Apply velocity changes
    for (idx, delta) in velocity_deltas.iter().enumerate() {
        self.particles.list[idx].velocity += *delta;
    }
}
```

**Benefits**:
- Smoother visual appearance (no "popping")
- Works better with sub-stepping
- Particles "flow" instead of "snap"

## Acceptance Criteria

### Pressure Solver
- [ ] Implement `solve_pressure_rbgs()` in `grid.rs`
- [ ] Remove temporary buffer allocation
- [ ] Reduce iterations from 80 to 40 while maintaining quality
- [ ] Add helper method `update_pressure_cell()`
- [ ] Verify incompressibility visually (water shouldn't compress)

### Collision Detection
- [ ] Add `clamp_velocities()` after G2P transfer in `flip.rs`
- [ ] Implement sub-stepped advection for fast particles
- [ ] No particles should tunnel through single-cell walls
- [ ] Test with high-velocity water stream against thin barriers

### Spatial Hash
- [ ] Add `cell_head` and `particle_next` fields to `FlipSimulation`
- [ ] Implement `build_spatial_hash()` - zero allocation rebuild
- [ ] Implement `for_each_neighbor()` helper
- [ ] Remove HashMap usage from particle separation

### Sub-Stepping
- [ ] Add `update_substep()` internal method
- [ ] Modify `update()` to run 4 substeps
- [ ] Reduce pressure iterations per substep (80 total → 20×4 = 80, or less)
- [ ] Verify stability improvement visually

### Particle Separation
- [ ] Implement `separate_particles_soft()` with force-based approach
- [ ] Tune stiffness parameter for smooth flow
- [ ] Remove old hard position correction method

### Testing & Verification
- [ ] Add unit test for pressure solver convergence
- [ ] Add test for collision detection with fast particle
- [ ] **FPS Target**: 60+ FPS with 2500 particles
- [ ] **Visual Target**: Water settles calmly without vibrating

## Technical Considerations

### Performance
- Red-Black GS: Same ops per iteration, but 2x fewer iterations needed
- Sub-stepping: Only triggers for fast particles (most frames: 0 substeps)
- Memory: Eliminate temp buffer saves ~65KB allocation per frame (128×128×4 bytes)

### Risks
- **Over-damping**: Velocity clamping might make water feel "slow" - tune threshold carefully
- **Solver instability**: GS can diverge if relaxation factor > 1 - stick with 0.25 (standard)
- **Sub-step overhead**: If water is consistently fast, sub-stepping adds cost

## Files to Modify

| File | Changes |
|------|---------|
| `crates/sim/src/grid.rs` | Replace `solve_pressure()` with `solve_pressure_rbgs()`, add `update_pressure_cell()` |
| `crates/sim/src/flip.rs` | Add spatial hash fields, `build_spatial_hash()`, `for_each_neighbor()`, `separate_particles_soft()`, sub-stepping, `clamp_velocities()` |

## References

### Internal
- Current solver: `crates/sim/src/grid.rs:242-286`
- Current collision: `crates/sim/src/flip.rs:308-387`
- Simulation loop: `crates/sim/src/flip.rs:33-71`

### External
- [Red-Black Gauss-Seidel explanation](https://en.wikipedia.org/wiki/Gauss–Seidel_method)
- [CFL Condition for fluid stability](https://en.wikipedia.org/wiki/Courant–Friedrichs–Lewy_condition)
- [Bridson's Fluid Simulation for Computer Graphics](https://www.cs.ubc.ca/~rbridson/fluidsimulation/)
- Research saved to: `ADVANCED_FLUID_SIMULATION_RESEARCH.md`, `RESEARCH_DOCUMENTATION.md`

## Implementation Order

1. **Red-Black GS** (highest impact on solver, lowest risk)
2. **Linked-Cell List** (biggest FPS improvement - eliminate allocations)
3. **Sub-stepping** (stability + natural tunneling prevention)
4. **Soft Particle Separation** (visual quality - smooth flow)
5. **Velocity clamping** (safety net for edge cases)
6. **Ray-based collision** (optional, only if tunneling persists)
