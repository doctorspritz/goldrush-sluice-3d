# Fluid Simulation Research: Practical Game-Applicable Techniques

**Research Date:** December 19, 2025
**Focus:** Jos Stam's Stable Fluids, Hybrid Lagrangian-Eulerian Methods, Sediment Transport, Fast Pressure Solvers

---

## Table of Contents
1. [Jos Stam's Stable Fluids - Pressure Projection](#1-jos-stams-stable-fluids---pressure-projection)
2. [Hybrid Lagrangian-Eulerian Methods (PIC/FLIP)](#2-hybrid-lagrangian-eulerian-methods-picflip)
3. [Sediment Transport Models](#3-sediment-transport-models)
4. [Fast Pressure Solvers for Real-Time Water Leveling](#4-fast-pressure-solvers-for-real-time-water-leveling)
5. [Practical Implementation Guide](#5-practical-implementation-guide)
6. [Code Resources and References](#6-code-resources-and-references)

---

## 1. Jos Stam's Stable Fluids - Pressure Projection

### Overview
Jos Stam's "Stable Fluids" algorithm (SIGGRAPH '99) is the foundation of most real-time fluid simulators. It solves the incompressible Euler equations, focusing on two fundamental parts: **Advection** and **Projection**.

### Key Properties
- **Unconditionally stable** - won't blow up regardless of timestep
- **Very easy to implement** - suitable for games and real-time applications
- **Focuses on visual plausibility** over physical accuracy

### The Pressure Projection Step

The projection step enforces incompressibility by ensuring the velocity field is divergence-free. This requires solving a **Poisson equation for pressure**.

#### How It Works:
1. After advection and force application, the velocity field may have non-zero divergence
2. The projection step computes a pressure field that, when its gradient is subtracted from the velocity, produces a divergence-free field
3. This creates the characteristic "leveling" behavior in fluids

**Mathematical Concept:**
```
v_incompressible = v - ∇p
```

Where:
- `v` is the current velocity field
- `p` is the pressure field (solution to Poisson equation)
- `∇p` is the pressure gradient
- `v_incompressible` is the resulting divergence-free velocity field

### Pressure-Based Horizontal Spreading

The pressure projection naturally creates horizontal spreading/leveling:
- High fluid columns create high pressure
- Pressure gradients push fluid from high to low regions
- This happens **automatically** through the projection step
- No additional "leveling" logic needed

### Key Insight for Implementation
The quality of your simulation depends heavily on how well you solve the Poisson equation. Poor solvers lead to:
- Residual divergence (compression/expansion artifacts)
- Need for compensatory hacks (vorticity confinement, MacCormack advection)
- Unrealistic damping and smearing

**References:**
- [Stable Fluids (Original Paper)](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/ns.pdf)
- [Stable Fluids - ResearchGate](https://www.researchgate.net/publication/2486965_Stable_Fluids)
- [Real-Time Fluid Dynamics for Games](http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf)

---

## 2. Hybrid Lagrangian-Eulerian Methods (PIC/FLIP)

### Why Hybrid Methods?

Hybrid methods combine the best of both worlds:
- **Particles (Lagrangian)**: Track material, maintain detail, handle splashing
- **Grid (Eulerian)**: Fast pressure solving, stable incompressibility enforcement

**Performance:** Hybrid methods can simulate several million particles vs. 60,000 in pure Lagrangian approaches.

### PIC (Particle-in-Cell) - 1963

**Process:**
1. Transfer particle velocities to grid (weighted averaging)
2. Solve forces and pressure on grid
3. Interpolate new grid velocities back to particles
4. Advect particles using updated velocities

**Problem:** Heavy numerical dissipation - fluids become overly viscous and damped.

### FLIP (Fluid-Implicit-Particle) - 1986/2005

**Key Innovation:** Transfer velocity **changes** instead of absolute values.

**Process:**
1. Transfer particle velocities to grid
2. Store old grid velocities
3. Solve forces and pressure on grid
4. Compute velocity change: `Δv = v_new_grid - v_old_grid`
5. Update particles: `v_particle += Δv` (not `v_particle = v_grid`)
6. Advect particles

**Advantages:**
- Eliminates dissipation from repeated interpolation
- Preserves fine-scale motion and detail
- Energy conservation

**Problem:** Can introduce noise and instability.

### PIC/FLIP Blending - Best Practice

The optimal approach is to **blend** PIC and FLIP:

```
v_particle_new = α * v_PIC + (1 - α) * v_FLIP
```

Where:
- `α = 0.05` to `0.1` is common (5-10% PIC, 90-95% FLIP)
- PIC provides stability
- FLIP provides energy preservation

### Particle-Grid Coupling Details

#### Particle-to-Grid Transfer
- Use **weighted averaging** with bilinear/trilinear hat function
- Each particle contributes to nearby grid cells (up to 27 in 3D)
- Accumulate velocities and weights at each grid point
- Use **MAC (Marker-and-Cell) grid** - staggered velocity components

**Pseudo-code:**
```rust
// For each particle
for particle in particles {
    let grid_pos = particle.position / grid_spacing;

    // Find neighboring cells
    for neighbor in get_neighbors(grid_pos, radius=2) {
        let weight = hat_kernel(distance(particle, neighbor));
        grid[neighbor].velocity_sum += particle.velocity * weight;
        grid[neighbor].weight_sum += weight;
    }
}

// Normalize
for cell in grid {
    cell.velocity = cell.velocity_sum / cell.weight_sum;
}
```

#### Grid-to-Particle Transfer
- **Much simpler** than particle-to-grid
- Use **trilinear interpolation** of 8 neighboring grid points
- Interpolate each velocity component (u, v, w) separately

**Pseudo-code:**
```rust
// For each particle
for particle in particles {
    let grid_pos = particle.position / grid_spacing;
    let i = floor(grid_pos);
    let f = fract(grid_pos); // fractional part

    // Trilinear interpolation
    let v_grid = interpolate_trilinear(grid_velocities, i, f);

    // PIC
    let v_pic = v_grid;

    // FLIP
    let delta_v = v_grid - particle.old_grid_velocity;
    let v_flip = particle.velocity + delta_v;

    // Blend
    particle.velocity = alpha * v_pic + (1.0 - alpha) * v_flip;
    particle.old_grid_velocity = v_grid; // store for next frame
}
```

### APIC (Affine Particle-in-Cell)

A newer alternative that represents particle velocities as **locally affine** (linear) rather than constant:
- Reduces dissipation without FLIP's noise
- Better angular momentum conservation
- More complex implementation

**References:**
- [Fluid Simulation Using Implicit Particles](http://www.danenglesson.com/images/portfolio/FLIP/rapport.pdf)
- [PIC/FLIP Fluid simulation](http://danenglesson.com/flipfluid.html)
- [WebGL-PIC-FLIP-Fluid GitHub](https://github.com/austinEng/WebGL-PIC-FLIP-Fluid)
- [Hybrid Particle-Grid Water Simulation using Multigrid](https://www.diva-portal.org/smash/get/diva2:708210/FULLTEXT01.pdf)
- [Affine Particle in Cell Method for MAC Grids](https://www.cs.ucr.edu/~craigs/papers/2019-mac-apic/paper.pdf)
- [2D Fluid Sim GitHub](https://github.com/davrempe/2d-fluid-sim)
- [Programming a Particle Fluid Simulation](https://raphaellaroca.wordpress.com/2025/05/02/programming-a-particle-fluid-simulation-part-1/)

---

## 3. Sediment Transport Models

### Overview
Sediment transport combines fluid simulation with particle dynamics for erosion, suspension, and deposition.

### Physical Parameters

#### Rouse Number (Suspension Criterion)
The **Rouse number** determines transport mode:

```
P = ω_s / (κ * u*)
```

Where:
- `ω_s` = particle settling velocity (downward)
- `u*` = shear velocity (turbulent mixing strength)
- `κ` = von Kármán constant (≈ 0.4)

**Transport Modes:**
- `P ≈ 0`: **Wash load** - fully suspended (settling << mixing)
- `P ≈ 1`: **Suspended load** - particles drift up and down
- `P > 2.5`: **Bed load** - particles roll/bounce along bottom

**Decision Logic for Games:**
```rust
let rouse_number = settling_velocity / (0.4 * shear_velocity);

if rouse_number < 0.8 {
    // Fully suspended - particle moves with fluid
    particle.state = Suspended;
} else if rouse_number < 2.5 {
    // Partially suspended - some settling
    particle.state = Settling;
} else {
    // Bed load - particle on bottom
    particle.state = BedLoad;
}
```

#### Shields Parameter (Entrainment Criterion)
The **Shields parameter** predicts when sediment begins to move:

```
θ = τ_b / ((ρ_sed - ρ_fluid) * g * D_sed)
```

Where:
- `τ_b` = bed shear stress
- `ρ_sed` = sediment density
- `ρ_fluid` = fluid density
- `g` = gravity
- `D_sed` = sediment grain diameter

**Critical value:** `θ_c ≈ 0.05` for most sediments

**Decision Logic:**
```rust
let shields = bed_shear_stress / ((sediment_density - fluid_density) * gravity * grain_diameter);

if shields > SHIELDS_CRITICAL {
    // Entrain sediment from bed
    lift_particle_from_bed();
}
```

#### Settling Velocity (Stokes Law)
For small particles, settling velocity is:

```
ω_s = ((ρ_sed - ρ_fluid) * g * d²) / (18 * μ)
```

Where:
- `d` = particle diameter
- `μ` = fluid dynamic viscosity

**Simplified for games:**
```rust
let settling_velocity = particle_size.powi(2) * GRAVITY * DENSITY_RATIO / VISCOSITY;
particle.velocity.y -= settling_velocity * dt;
```

### Game Implementation Strategy

#### 1. Particle States
```rust
enum SedimentState {
    Suspended,  // moves with fluid
    Settling,   // gradual fall
    BedLoad,    // rolling on bottom
    Deposited,  // stationary on bed
}
```

#### 2. Update Loop
```rust
fn update_sediment(particle: &mut Particle, fluid_velocity: Vec3, dt: f32) {
    match particle.state {
        Suspended => {
            // Move with fluid + small settling
            particle.velocity = fluid_velocity;
            particle.velocity.y -= settling_velocity * 0.1 * dt;

            // Check if should start settling
            if fluid_velocity.length() < SUSPENSION_THRESHOLD {
                particle.state = Settling;
            }
        },

        Settling => {
            // Influenced by fluid but also settling
            particle.velocity = lerp(particle.velocity, fluid_velocity, 0.5);
            particle.velocity.y -= settling_velocity * dt;

            // Check if reached bed
            if particle.position.y <= bed_height {
                particle.state = Deposited;
            }
        },

        BedLoad => {
            // Slide/roll along bottom
            let horizontal_velocity = fluid_velocity.xz();
            if horizontal_velocity.length() > BED_LOAD_THRESHOLD {
                particle.velocity.xz = horizontal_velocity * 0.3;
            }
        },

        Deposited => {
            // Check if should be re-entrained
            let shields = calculate_shields_parameter(particle);
            if shields > SHIELDS_CRITICAL {
                particle.state = Suspended;
                particle.velocity = fluid_velocity;
            }
        }
    }
}
```

#### 3. Capacity-Based Erosion/Deposition
```rust
let sediment_capacity = calculate_capacity(fluid_velocity, fluid_depth);

if current_sediment < sediment_capacity {
    // Erode terrain
    let erosion_amount = (sediment_capacity - current_sediment) * EROSION_RATE * dt;
    terrain_height -= erosion_amount;
    current_sediment += erosion_amount;
} else {
    // Deposit sediment
    let deposition_amount = (current_sediment - sediment_capacity) * DEPOSITION_RATE * dt;
    terrain_height += deposition_amount;
    current_sediment -= deposition_amount;
}
```

**References:**
- [Sediment Transport - Wikipedia](https://en.wikipedia.org/wiki/Sediment_transport)
- [Rouse Number - Wikipedia](https://en.wikipedia.org/wiki/Rouse_number)
- [Sediment Transport Model - FLOW-3D](https://www.flow3d.com/modeling-capabilities/sediment-transport-model/)
- [Interactive Hydraulic Erosion Simulator](https://huw-man.github.io/Interactive-Erosion-Simulator-on-GPU/)
- [Simulating Hydraulic Erosion of Terrain](https://gameidea.org/2023/12/22/simulating-hydraulic-erosion-of-terrain/)
- [Terrain Erosion 3 Ways - GitHub](https://github.com/dandrino/terrain-erosion-3-ways)

---

## 4. Fast Pressure Solvers for Real-Time Water Leveling

### The Core Problem
Pressure projection requires solving a **Poisson equation**:

```
∇²p = ∇·v
```

This is the bottleneck of real-time fluid simulation.

### Solver Comparison

#### Jacobi Method
**Simple but inefficient:**
- Easy to implement
- Parallelizes well (good for GPU)
- **Very slow convergence** - needs 40-80 iterations minimum, 1000+ for quality

**When to use:** Prototyping, simple simulations, when you can't implement multigrid

**Iteration formula:**
```rust
// For each cell
p_new[i,j,k] = (p[i-1,j,k] + p[i+1,j,k] +
                p[i,j-1,k] + p[i,j+1,k] +
                p[i,j,k-1] + p[i,j,k+1] -
                divergence[i,j,k] * dx²) / 6.0;
```

**Critical:** Must use separate read/write buffers - don't update in-place.

#### Gauss-Seidel Method
**Better convergence:**
- Converges **2x faster** than Jacobi
- Uses latest values immediately
- **Red-Black variant** parallelizes well

**When to use:** When you can't implement multigrid but need better performance

**Red-Black pattern:**
```rust
// Red cells (checkerboard pattern)
for (i, j, k) in red_cells {
    p[i,j,k] = (p[i-1,j,k] + p[i+1,j,k] + ...) / 6.0;
}

// Black cells
for (i, j, k) in black_cells {
    p[i,j,k] = (p[i-1,j,k] + p[i+1,j,k] + ...) / 6.0;
}
```

#### Multigrid Method
**The gold standard:**
- **1 V-cycle ≈ 1000 Jacobi iterations**
- **Cost ≈ 10 Jacobi iterations**
- 4 V-cycles gives excellent results
- Much more complex to implement

**Key idea:**
- Solve on multiple grid resolutions
- Coarse grids eliminate low-frequency error
- Fine grids handle high-frequency detail

**When to use:** Production-quality simulations, when performance matters

**V-Cycle structure:**
```
Fine grid (full resolution)
  ↓ Smooth (few Jacobi/GS iterations)
  ↓ Restrict (downsample to coarser grid)
Medium grid
  ↓ Smooth
  ↓ Restrict
Coarse grid
  ↓ Solve exactly or smooth heavily
  ↑ Prolongate (upsample)
Medium grid
  ↑ Smooth
  ↑ Prolongate
Fine grid (full resolution)
  ↑ Smooth
```

### Practical Recommendations

**For prototyping:**
```rust
// Jacobi with 40-80 iterations
for _ in 0..60 {
    jacobi_iteration(&mut pressure, &divergence);
}
```

**For production (medium quality):**
```rust
// Red-Black Gauss-Seidel with 20-40 iterations
for _ in 0..30 {
    gauss_seidel_red(&mut pressure, &divergence);
    gauss_seidel_black(&mut pressure, &divergence);
}
```

**For production (high quality):**
```rust
// Multigrid with 3-5 V-cycles
for _ in 0..4 {
    multigrid_v_cycle(&mut pressure, &divergence);
}
```

### Shallow Water Equations (Height Field Alternative)

For **large-scale water** where vertical motion isn't important, use **Shallow Water Equations** instead:

**Advantages:**
- 2D instead of 3D (much faster)
- Height field representation
- Natural pressure-based spreading
- No explicit pressure solve needed

**Equation:**
```
∂h/∂t + ∇·(hv) = 0                    // Mass conservation
∂v/∂t + v·∇v = -g∇h + friction        // Momentum
```

**Implementation:**
```rust
struct WaterCell {
    height: f32,        // Water column height
    velocity: Vec2,     // Horizontal velocity
    terrain: f32,       // Bed elevation
}

fn shallow_water_step(grid: &mut Grid2D, dt: f32) {
    // 1. Compute fluxes between cells
    for cell in grid.cells() {
        let surface_level = cell.terrain + cell.height;

        for neighbor in cell.neighbors() {
            let neighbor_level = neighbor.terrain + neighbor.height;
            let height_diff = surface_level - neighbor_level;

            // Pressure gradient drives flow
            let flux = GRAVITY * height_diff * dt;
            cell.flux_to_neighbor = flux;
        }
    }

    // 2. Update heights and velocities
    for cell in grid.cells() {
        let net_flux = cell.incoming_flux - cell.outgoing_flux;
        cell.height += net_flux * dt;
        cell.velocity = cell.outgoing_flux / cell.height;
    }
}
```

**When to use:** Rivers, lakes, flooding, large-scale water where splashing isn't important

**References:**
- [Realtime Fluid Simulation: Projection - GitHub Gist](https://gist.github.com/vassvik/f06a453c18eae03a9ad4dc8cc011d2dc)
- [3D Parallel Multigrid Methods for Real-Time Fluid Simulation](https://www.researchgate.net/publication/323157448_3D_Parallel_Multigrid_Methods_for_Real-Time_Fluid_Simulation)
- [Fast Fluid Dynamics Simulation on GPU - NVIDIA](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)
- [A Parallel Multigrid Poisson Solver](https://www.researchgate.net/publication/220789182_A_Parallel_Multigrid_Poisson_Solver_for_Fluids_Simulation_on_Large_Grids)
- [Shallow Water Equations - Wikipedia](https://en.wikipedia.org/wiki/Shallow_water_equations)
- [Real-time Simulation of Large Bodies of Water](https://matthias-research.github.io/pages/publications/hfFluid.pdf)
- [Height-Field-Water GitHub](https://github.com/ucanbizon/Height-Field-Water)
- [Real-time Height-field Simulation of Sand and Water Mixtures](https://kuiwuchn.github.io/RTWaterAndSand.pdf)

---

## 5. Practical Implementation Guide

### Recommended Architecture for Particle Fluid with Sediment

```rust
struct FluidSimulation {
    // Grid-based
    grid: MAC_Grid,              // Staggered velocity grid
    pressure: Grid3D<f32>,       // Pressure field
    divergence: Grid3D<f32>,     // Velocity divergence

    // Particle-based
    fluid_particles: Vec<Particle>,
    sediment_particles: Vec<SedimentParticle>,

    // Configuration
    cell_size: f32,
    alpha: f32,                  // PIC/FLIP blend (0.05 - 0.1)
}

struct Particle {
    position: Vec3,
    velocity: Vec3,
    old_grid_velocity: Vec3,     // For FLIP
}

struct SedimentParticle {
    position: Vec3,
    velocity: Vec3,
    state: SedimentState,
    size: f32,
    density: f32,
}
```

### Simulation Step

```rust
fn simulation_step(&mut self, dt: f32) {
    // 1. Clear grid
    self.grid.clear();

    // 2. Particle-to-Grid (P2G)
    self.transfer_particles_to_grid();

    // 3. Store old velocities (for FLIP)
    let old_velocities = self.grid.velocities.clone();

    // 4. Apply forces (gravity, etc.)
    self.grid.apply_gravity(dt);

    // 5. PRESSURE PROJECTION (the key step)
    self.compute_divergence();
    self.solve_pressure();           // Jacobi/GS/Multigrid
    self.apply_pressure_gradient(dt);

    // 6. Grid-to-Particle (G2P) with PIC/FLIP blend
    self.transfer_grid_to_particles(&old_velocities);

    // 7. Advect particles
    for particle in &mut self.fluid_particles {
        particle.position += particle.velocity * dt;
    }

    // 8. Update sediment
    self.update_sediment(dt);

    // 9. Handle boundaries and collisions
    self.enforce_boundaries();
}
```

### Pressure Projection Detail

```rust
fn solve_pressure(&mut self) {
    // Option A: Jacobi (simple but slow)
    for _ in 0..60 {
        self.jacobi_iteration();
    }

    // Option B: Red-Black Gauss-Seidel (better)
    for _ in 0..30 {
        self.gauss_seidel_iteration(RedBlack::Red);
        self.gauss_seidel_iteration(RedBlack::Black);
    }

    // Option C: Multigrid (best)
    for _ in 0..4 {
        self.multigrid_v_cycle();
    }
}

fn apply_pressure_gradient(&mut self, dt: f32) {
    // Subtract pressure gradient from velocity
    for i in 1..self.grid.width-1 {
        for j in 1..self.grid.height-1 {
            for k in 1..self.grid.depth-1 {
                // Compute pressure gradient
                let grad_x = (self.pressure[i+1,j,k] - self.pressure[i-1,j,k]) / (2.0 * self.cell_size);
                let grad_y = (self.pressure[i,j+1,k] - self.pressure[i,j-1,k]) / (2.0 * self.cell_size);
                let grad_z = (self.pressure[i,j,k+1] - self.pressure[i,j,k-1]) / (2.0 * self.cell_size);

                // Update velocity (makes it divergence-free)
                self.grid.u[i,j,k] -= grad_x * dt;
                self.grid.v[i,j,k] -= grad_y * dt;
                self.grid.w[i,j,k] -= grad_z * dt;
            }
        }
    }
}
```

### Sediment Coupling

```rust
fn update_sediment(&mut self, dt: f32) {
    for particle in &mut self.sediment_particles {
        // Get fluid velocity at particle position
        let fluid_vel = self.grid.sample_velocity(particle.position);

        // Compute suspension criteria
        let settling_vel = self.compute_settling_velocity(particle);
        let shear_vel = self.compute_shear_velocity(particle.position);
        let rouse = settling_vel / (0.4 * shear_vel);

        match particle.state {
            SedimentState::Suspended if rouse > 2.5 => {
                particle.state = SedimentState::Settling;
            },
            SedimentState::Settling if rouse < 0.8 => {
                particle.state = SedimentState::Suspended;
            },
            _ => {}
        }

        // Update based on state
        match particle.state {
            SedimentState::Suspended => {
                particle.velocity = fluid_vel;
                particle.velocity.y -= settling_vel * 0.1 * dt;
            },
            SedimentState::Settling => {
                particle.velocity = lerp(particle.velocity, fluid_vel, 0.5);
                particle.velocity.y -= settling_vel * dt;
            },
            _ => {}
        }

        // Advect
        particle.position += particle.velocity * dt;

        // Check boundaries
        if particle.position.y <= self.terrain_height(particle.position.xz()) {
            particle.state = SedimentState::Deposited;
            self.deposit_sediment(particle);
        }
    }
}
```

---

## 6. Code Resources and References

### Official Papers
- [Stable Fluids - Jos Stam (1999)](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/ns.pdf) - The foundational paper
- [Real-Time Fluid Dynamics for Games - Jos Stam](http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf) - Game-focused version

### PIC/FLIP Resources
- [Fluid Simulation Using Implicit Particles (PDF)](http://www.danenglesson.com/images/portfolio/FLIP/rapport.pdf) - Excellent FLIP tutorial
- [Affine Particle in Cell for MAC Grids (PDF)](https://www.cs.ucr.edu/~craigs/papers/2019-mac-apic/paper.pdf) - APIC method
- [Hybrid Particle-Grid Water Simulation (PDF)](https://www.diva-portal.org/smash/get/diva2:708210/FULLTEXT01.pdf) - Multigrid with FLIP

### Open Source Implementations
- [WebGL-PIC-FLIP-Fluid](https://github.com/austinEng/WebGL-PIC-FLIP-Fluid) - WebGL implementation
- [2D Fluid Sim](https://github.com/davrempe/2d-fluid-sim) - 2D PIC/FLIP in Python
- [StableFluids](https://github.com/albelax/StableFluids) - Parallel implementation
- [Terrain Erosion 3 Ways](https://github.com/dandrino/terrain-erosion-3-ways) - Erosion examples
- [Height-Field-Water](https://github.com/ucanbizon/Height-Field-Water) - Height field approach

### Interactive Tools
- [Interactive Hydraulic Erosion Simulator](https://huw-man.github.io/Interactive-Erosion-Simulator-on-GPU/) - GPU-based erosion
- [PIC/FLIP Fluid Simulation Demo](http://danenglesson.com/flipfluid.html) - Interactive demo

### Tutorials and Guides
- [Programming a Particle Fluid Simulation](https://raphaellaroca.wordpress.com/2025/05/02/programming-a-particle-fluid-simulation-part-1/) - Step-by-step tutorial
- [Simulating Hydraulic Erosion of Terrain](https://gameidea.org/2023/12/22/simulating-hydraulic-erosion-of-terrain/) - Game erosion
- [Realtime Fluid Simulation: Projection](https://gist.github.com/vassvik/f06a453c18eae03a9ad4dc8cc011d2dc) - Projection step explained

### Advanced Topics
- [Fast Fluid Dynamics Simulation on GPU - NVIDIA](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)
- [3D Parallel Multigrid Methods](https://www.researchgate.net/publication/323157448_3D_Parallel_Multigrid_Methods_for_Real-Time_Fluid_Simulation)
- [Real-time Height-field Sand and Water](https://kuiwuchn.github.io/RTWaterAndSand.pdf) - SIGGRAPH 2023

### Scientific Resources
- [Sediment Transport Model - FLOW-3D](https://www.flow3d.com/modeling-capabilities/sediment-transport-model/)
- [Shallow Water Equations - Wikipedia](https://en.wikipedia.org/wiki/Shallow_water_equations)
- [Rouse Number - Wikipedia](https://en.wikipedia.org/wiki/Rouse_number)
- [Sediment Transport - Wikipedia](https://en.wikipedia.org/wiki/Sediment_transport)

---

## Key Takeaways for Game Development

1. **Use PIC/FLIP hybrid** for particles + grid coupling
   - 90-95% FLIP, 5-10% PIC for stability
   - Particles for detail, grid for pressure

2. **Pressure projection is critical**
   - This creates water leveling automatically
   - Better solvers = better results
   - Start with Jacobi, upgrade to multigrid

3. **Sediment uses simple criteria**
   - Rouse number for suspension (velocity vs settling)
   - Shields parameter for entrainment (shear stress)
   - Capacity-based erosion/deposition

4. **For large-scale water, use height fields**
   - Shallow Water Equations
   - 2D instead of 3D
   - Natural pressure-based spreading

5. **Prioritize visual plausibility over accuracy**
   - Games need fast, stable, controllable simulations
   - Physical correctness is secondary to good visuals
   - Tune parameters for gameplay, not reality

---

**Document Version:** 1.0
**Last Updated:** December 19, 2025
