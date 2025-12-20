# PIC/FLIP Sluice Simulation - Vortex Formation

## Status: IMPLEMENTED ✓

**Implementation Date:** 2025-12-20

The PIC/FLIP simulation has been implemented with the following files:
- `crates/sim/src/particle.rs` - Particle struct (water/mud)
- `crates/sim/src/grid.rs` - MAC grid with pressure solver
- `crates/sim/src/flip.rs` - PIC/FLIP transfer and simulation loop
- `crates/sim/src/sluice.rs` - Sluice geometry setup
- `crates/game/src/main.rs` - Rendering and controls

**Controls:**
- `Space` - Pause/Resume
- `V` - Toggle velocity field visualization
- `M` - Toggle mud/water spawning
- `R` - Reset simulation
- `C` - Clear particles
- `Left Click` - Spawn particles at cursor

## Goal

Two particle types (water, mud) flowing over a sluice with riffles. Vortices form behind riffles naturally through pressure projection. No CA, no swapping.

## Core Concept

From FLUID_SIMULATION_RESEARCH.md:

> **PIC/FLIP hybrid** for particles + grid coupling
> - Particles for detail, grid for pressure
> - Pressure projection creates vortices automatically

## Data Structures

```rust
/// A fluid particle - water or mud
struct Particle {
    position: Vec2,           // Continuous position in world
    velocity: Vec2,           // Current velocity
    old_grid_velocity: Vec2,  // For FLIP delta calculation
    density: f32,             // Water: 1.0, Mud: 2.5
}

/// Staggered MAC grid for velocity/pressure
struct Grid {
    width: usize,
    height: usize,
    cell_size: f32,

    // Velocity components (staggered - u on left edges, v on bottom edges)
    u: Vec<f32>,              // Horizontal velocity (width+1 × height)
    v: Vec<f32>,              // Vertical velocity (width × height+1)

    // Pressure (cell centers)
    pressure: Vec<f32>,       // (width × height)
    divergence: Vec<f32>,     // (width × height)

    // Cell type for boundary handling
    cell_type: Vec<CellType>, // Solid (rock), Fluid, Air
}

enum CellType {
    Solid,  // Rock/riffles - blocks flow
    Fluid,  // Contains particles
    Air,    // Empty
}
```

## Algorithm: Each Frame

### Step 1: Classify Cells

```rust
fn classify_cells(grid: &mut Grid, particles: &[Particle]) {
    // Reset to air
    for cell in &mut grid.cell_type {
        *cell = CellType::Air;
    }

    // Mark solid cells (terrain/riffles) - set from level geometry
    for (i, is_solid) in terrain.iter().enumerate() {
        if *is_solid {
            grid.cell_type[i] = CellType::Solid;
        }
    }

    // Mark fluid cells (contain particles)
    for particle in particles {
        let cell = grid.pos_to_cell(particle.position);
        if grid.cell_type[cell] != CellType::Solid {
            grid.cell_type[cell] = CellType::Fluid;
        }
    }
}
```

### Step 2: Particle → Grid (P2G)

Transfer particle velocities to grid using weighted interpolation.

```rust
fn particles_to_grid(grid: &mut Grid, particles: &[Particle]) {
    // Clear accumulators
    let mut u_sum = vec![0.0; grid.u.len()];
    let mut u_weight = vec![0.0; grid.u.len()];
    let mut v_sum = vec![0.0; grid.v.len()];
    let mut v_weight = vec![0.0; grid.v.len()];

    for particle in particles {
        // U component (staggered - sample at left edges)
        let u_pos = particle.position - vec2(grid.cell_size * 0.5, 0.0);
        let (i, j, weights) = grid.get_interp_weights(u_pos);

        for (di, dj, w) in weights {
            let idx = grid.u_index(i + di, j + dj);
            u_sum[idx] += particle.velocity.x * w;
            u_weight[idx] += w;
        }

        // V component (staggered - sample at bottom edges)
        let v_pos = particle.position - vec2(0.0, grid.cell_size * 0.5);
        let (i, j, weights) = grid.get_interp_weights(v_pos);

        for (di, dj, w) in weights {
            let idx = grid.v_index(i + di, j + dj);
            v_sum[idx] += particle.velocity.y * w;
            v_weight[idx] += w;
        }
    }

    // Normalize
    for i in 0..grid.u.len() {
        if u_weight[i] > 0.0 {
            grid.u[i] = u_sum[i] / u_weight[i];
        }
    }
    for i in 0..grid.v.len() {
        if v_weight[i] > 0.0 {
            grid.v[i] = v_sum[i] / v_weight[i];
        }
    }
}
```

### Step 3: Store Old Grid Velocities (for FLIP)

```rust
fn store_old_velocities(particles: &mut [Particle], grid: &Grid) {
    for particle in particles {
        particle.old_grid_velocity = grid.sample_velocity(particle.position);
    }
}
```

### Step 4: Apply External Forces

```rust
fn apply_forces(grid: &mut Grid, dt: f32) {
    const GRAVITY: f32 = 9.8;

    // Add gravity to vertical velocity
    for v in &mut grid.v {
        *v -= GRAVITY * dt;
    }
}
```

### Step 4b: Vorticity Confinement (maintains swirling)

From PavelDoGreat/WebGL-Fluid-Simulation - this keeps vortices from dissipating.

```rust
fn vorticity_confinement(grid: &mut Grid, dt: f32) {
    const CURL_STRENGTH: f32 = 30.0;

    // First pass: compute curl (vorticity) at each cell
    let mut curl = vec![0.0; grid.width * grid.height];

    for j in 1..grid.height - 1 {
        for i in 1..grid.width - 1 {
            let idx = grid.cell_index(i, j);

            // Curl = dv/dx - du/dy
            let du_dy = (grid.u[grid.u_index(i, j + 1)] - grid.u[grid.u_index(i, j - 1)]) * 0.5;
            let dv_dx = (grid.v[grid.v_index(i + 1, j)] - grid.v[grid.v_index(i - 1, j)]) * 0.5;

            curl[idx] = dv_dx - du_dy;
        }
    }

    // Second pass: apply vorticity force
    for j in 2..grid.height - 2 {
        for i in 2..grid.width - 2 {
            let idx = grid.cell_index(i, j);

            if grid.cell_type[idx] != CellType::Fluid {
                continue;
            }

            // Gradient of curl magnitude
            let curl_l = curl[grid.cell_index(i - 1, j)].abs();
            let curl_r = curl[grid.cell_index(i + 1, j)].abs();
            let curl_b = curl[grid.cell_index(i, j - 1)].abs();
            let curl_t = curl[grid.cell_index(i, j + 1)].abs();

            let grad_x = (curl_r - curl_l) * 0.5;
            let grad_y = (curl_t - curl_b) * 0.5;

            let len = (grad_x * grad_x + grad_y * grad_y).sqrt() + 1e-5;
            let nx = grad_x / len;
            let ny = grad_y / len;

            // Force perpendicular to gradient, proportional to curl
            let c = curl[idx];
            let fx = ny * c * CURL_STRENGTH;
            let fy = -nx * c * CURL_STRENGTH;

            // Apply to velocity
            grid.u[grid.u_index(i, j)] += fx * dt;
            grid.v[grid.v_index(i, j)] += fy * dt;
        }
    }
}
```

### Step 5: Pressure Projection (THE KEY STEP)

This is where vortices form. The pressure solver enforces incompressibility, which creates the swirling motion behind obstacles.

```rust
fn pressure_projection(grid: &mut Grid, dt: f32) {
    // 5a: Compute divergence
    for j in 0..grid.height {
        for i in 0..grid.width {
            if grid.cell_type[grid.cell_index(i, j)] != CellType::Fluid {
                grid.divergence[grid.cell_index(i, j)] = 0.0;
                continue;
            }

            let u_right = grid.u[grid.u_index(i + 1, j)];
            let u_left = grid.u[grid.u_index(i, j)];
            let v_top = grid.v[grid.v_index(i, j + 1)];
            let v_bottom = grid.v[grid.v_index(i, j)];

            let div = (u_right - u_left + v_top - v_bottom) / grid.cell_size;
            grid.divergence[grid.cell_index(i, j)] = div;
        }
    }

    // 5b: Solve Poisson equation for pressure (Jacobi iterations)
    // From PavelDoGreat/WebGL-Fluid-Simulation:
    // p = (L + R + B + T - divergence) * 0.25
    let mut new_pressure = vec![0.0; grid.pressure.len()];

    for _ in 0..40 {  // 40 Jacobi iterations (20-60 typical)
        for j in 1..grid.height - 1 {
            for i in 1..grid.width - 1 {
                let idx = grid.cell_index(i, j);

                if grid.cell_type[idx] != CellType::Fluid {
                    new_pressure[idx] = 0.0;
                    continue;
                }

                // Neighbor pressures (treat solid as 0)
                let l = grid.pressure[grid.cell_index(i - 1, j)];
                let r = grid.pressure[grid.cell_index(i + 1, j)];
                let b = grid.pressure[grid.cell_index(i, j - 1)];
                let t = grid.pressure[grid.cell_index(i, j + 1)];
                let div = grid.divergence[idx];

                // Simple Jacobi step
                new_pressure[idx] = (l + r + b + t - div) * 0.25;
            }
        }

        std::mem::swap(&mut grid.pressure, &mut new_pressure);
    }

    // 5c: Subtract pressure gradient from velocity
    for j in 0..grid.height {
        for i in 1..grid.width {
            let idx_left = grid.cell_index(i - 1, j);
            let idx_right = grid.cell_index(i, j);

            // Only update if at least one side is fluid
            if grid.cell_type[idx_left] == CellType::Fluid ||
               grid.cell_type[idx_right] == CellType::Fluid {
                let grad = (grid.pressure[idx_right] - grid.pressure[idx_left]) / grid.cell_size;
                grid.u[grid.u_index(i, j)] -= grad * dt;
            }

            // Zero velocity at solid boundaries
            if grid.cell_type[idx_left] == CellType::Solid ||
               grid.cell_type[idx_right] == CellType::Solid {
                grid.u[grid.u_index(i, j)] = 0.0;
            }
        }
    }

    for j in 1..grid.height {
        for i in 0..grid.width {
            let idx_bottom = grid.cell_index(i, j - 1);
            let idx_top = grid.cell_index(i, j);

            if grid.cell_type[idx_bottom] == CellType::Fluid ||
               grid.cell_type[idx_top] == CellType::Fluid {
                let grad = (grid.pressure[idx_top] - grid.pressure[idx_bottom]) / grid.cell_size;
                grid.v[grid.v_index(i, j)] -= grad * dt;
            }

            if grid.cell_type[idx_bottom] == CellType::Solid ||
               grid.cell_type[idx_top] == CellType::Solid {
                grid.v[grid.v_index(i, j)] = 0.0;
            }
        }
    }
}
```

### Step 6: Grid → Particle (G2P) with PIC/FLIP Blend

```rust
fn grid_to_particles(particles: &mut [Particle], grid: &Grid) {
    const ALPHA: f32 = 0.05;  // 5% PIC, 95% FLIP

    for particle in particles {
        let v_grid = grid.sample_velocity(particle.position);

        // PIC: use grid velocity directly
        let v_pic = v_grid;

        // FLIP: add velocity change to particle
        let delta_v = v_grid - particle.old_grid_velocity;
        let v_flip = particle.velocity + delta_v;

        // Blend
        particle.velocity = ALPHA * v_pic + (1.0 - ALPHA) * v_flip;
    }
}
```

### Step 7: Advect Particles

```rust
fn advect_particles(particles: &mut [Particle], grid: &Grid, dt: f32) {
    for particle in particles {
        // Move with velocity
        particle.position += particle.velocity * dt;

        // Collision with solids - push out and zero normal velocity
        resolve_solid_collision(particle, grid);
    }
}
```

### Step 8: Density-Based Settling (Mud vs Water)

```rust
fn apply_settling(particles: &mut [Particle], dt: f32) {
    const WATER_DENSITY: f32 = 1.0;

    for particle in particles {
        if particle.density > WATER_DENSITY {
            // Settling velocity based on density difference
            // Stokes law simplified
            let settling = (particle.density - WATER_DENSITY) * 0.5;
            particle.velocity.y -= settling * dt;
        }
    }
}
```

## Complete Update Loop

```rust
fn update(particles: &mut Vec<Particle>, grid: &mut Grid, dt: f32) {
    // 1. Classify cells (solid/fluid/air)
    classify_cells(grid, particles);

    // 2. Transfer particle velocities to grid
    particles_to_grid(grid, particles);

    // 3. Store old velocities for FLIP
    store_old_velocities(particles, grid);

    // 4. Apply forces (gravity)
    apply_forces(grid, dt);

    // 5. PRESSURE PROJECTION - creates vortices!
    pressure_projection(grid, dt);

    // 6. Transfer grid velocities back to particles
    grid_to_particles(particles, grid);

    // 7. Apply density-based settling
    apply_settling(particles, dt);

    // 8. Advect particles
    advect_particles(particles, grid, dt);
}
```

## Why Vortices Form

When water flows over a riffle (solid obstacle):
1. Flow is blocked on the upstream side → high pressure
2. Flow separates on the downstream side → low pressure
3. Pressure gradient creates circulation
4. FLIP preserves this angular momentum → persistent vortex

The pressure projection step (Step 5) is where this happens automatically. No special vortex code needed.

## Sluice Setup

```rust
fn create_sluice(grid: &mut Grid) {
    // Floor
    for i in 0..grid.width {
        grid.cell_type[grid.cell_index(i, 0)] = CellType::Solid;
    }

    // Riffles (vertical bars every N cells)
    let riffle_spacing = 20;
    let riffle_height = 5;

    for x in (0..grid.width).step_by(riffle_spacing) {
        for y in 1..=riffle_height {
            grid.cell_type[grid.cell_index(x, y)] = CellType::Solid;
        }
    }
}

fn spawn_water(particles: &mut Vec<Particle>, x: f32, y: f32, count: usize) {
    for _ in 0..count {
        particles.push(Particle {
            position: vec2(x + rand(-1.0, 1.0), y + rand(-1.0, 1.0)),
            velocity: vec2(2.0, 0.0),  // Initial rightward flow
            old_grid_velocity: Vec2::ZERO,
            density: 1.0,  // Water
        });
    }
}

fn spawn_mud(particles: &mut Vec<Particle>, x: f32, y: f32, count: usize) {
    for _ in 0..count {
        particles.push(Particle {
            position: vec2(x + rand(-1.0, 1.0), y + rand(-1.0, 1.0)),
            velocity: vec2(2.0, 0.0),
            old_grid_velocity: Vec2::ZERO,
            density: 2.5,  // Mud - denser, settles
        });
    }
}
```

## Rendering

```rust
fn render(particles: &[Particle], grid: &Grid) {
    // Draw terrain/riffles
    for j in 0..grid.height {
        for i in 0..grid.width {
            if grid.cell_type[grid.cell_index(i, j)] == CellType::Solid {
                draw_rect(i, j, GRAY);
            }
        }
    }

    // Draw particles
    for particle in particles {
        let color = if particle.density > 1.5 {
            BROWN  // Mud
        } else {
            BLUE   // Water
        };
        draw_circle(particle.position.x, particle.position.y, 1.0, color);
    }

    // Optional: draw velocity field
    for j in 0..grid.height {
        for i in 0..grid.width {
            let vel = grid.sample_velocity_at_cell(i, j);
            draw_line(i, j, i + vel.x * 0.1, j + vel.y * 0.1, WHITE);
        }
    }
}
```

## Test Criteria

1. **Water flows over riffles** - continuous flow from left to right
2. **Vortices form behind riffles** - visible swirling in velocity field
3. **Mud settles** - ends up at the bottom, behind riffles where velocity is low
4. **Increasing flow = bigger vortices** - more particles = more momentum = larger eddies
5. **No explosions** - pressure solver keeps it stable

## Files to Create

| File | Purpose |
|------|---------|
| `crates/sim/src/particle.rs` | Particle struct |
| `crates/sim/src/grid.rs` | MAC grid, pressure solver |
| `crates/sim/src/flip.rs` | PIC/FLIP transfer functions |
| `crates/sim/src/sluice.rs` | Sluice geometry setup |

## References

- FLUID_SIMULATION_RESEARCH.md Section 2 (PIC/FLIP)
- FLUID_SIMULATION_RESEARCH.md Section 4 (Pressure Solvers)
- [Fluid Simulation Using Implicit Particles](http://www.danenglesson.com/images/portfolio/FLIP/rapport.pdf)
