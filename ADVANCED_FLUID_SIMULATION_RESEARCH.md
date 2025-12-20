# Advanced Fluid Simulation Best Practices Research

**Research Date:** December 20, 2025
**Focus:** Red-Black Gauss-Seidel vs Jacobi, Particle Collision Detection, Velocity Clamping, CCD for Particle Systems

This document provides deep-dive research into advanced fluid simulation techniques with focus on practical implementation patterns.

---

## Table of Contents
1. [Red-Black Gauss-Seidel vs Jacobi Iteration for Pressure Solving](#1-red-black-gauss-seidel-vs-jacobi-iteration)
2. [Particle Collision Detection in PIC/FLIP - Preventing Tunneling](#2-particle-collision-detection-in-picflip)
3. [Velocity Clamping and Sub-Stepping for Stable Simulation](#3-velocity-clamping-and-sub-stepping)
4. [Continuous Collision Detection (CCD) for Particle Systems](#4-continuous-collision-detection-ccd)
5. [Summary Recommendations](#5-summary-recommendations)
6. [Academic References](#6-academic-references)

---

## 1. Red-Black Gauss-Seidel vs Jacobi Iteration

### Overview

Both Jacobi and Gauss-Seidel methods are iterative solvers for the pressure Poisson equation in incompressible fluid simulation. The fundamental trade-off is between **convergence speed** (Gauss-Seidel wins) and **parallelizability** (Jacobi wins).

### Jacobi Method

**Strengths:**
- **Highly parallelizable**: All unknowns can be computed completely in parallel because new values don't depend on each other within an iteration
- **GPU-friendly**: Trivial to implement efficiently on GPUs due to independent calculations
- **Simple implementation**: Only requires reading old values, no race conditions
- **Two storage vectors**: Requires separate old and new state arrays

**Weaknesses:**
- **Slower convergence**: Requires significantly more iterations than Gauss-Seidel to reach the same accuracy
- **Memory bandwidth**: More memory reads/writes per iteration

**When to use:**
- GPU implementations where parallelism is critical
- When simplicity of implementation matters
- Real-time applications where consistent per-frame performance is more important than minimum iteration count

### Gauss-Seidel Method

**Strengths:**
- **Faster convergence**: Typically converges in fewer iterations than Jacobi (often 2x fewer or better)
- **Uses updated values immediately**: As soon as a new value is calculated, it's used for subsequent calculations in the same iteration
- **Single storage vector**: Can overwrite values in-place, advantageous for very large problems
- **Better for CPU**: More efficient on serial architectures

**Weaknesses:**
- **Highly serial algorithm**: Has a long critical path, making parallelization difficult
- **Poor GPU performance**: Direct implementation on GPU is inefficient
- **Memory access patterns**: Can have poor cache coherency

**When to use:**
- CPU-based simulations
- When iteration count is the bottleneck
- Offline/non-real-time applications where total convergence time matters more than per-iteration time

### Red-Black Gauss-Seidel (Best of Both Worlds)

**The Solution for GPU Parallelization:**

Red-Black Gauss-Seidel uses a checkerboard coloring scheme to parallelize Gauss-Seidel:

1. **Grid coloring**: Divide grid cells into "red" and "black" like a checkerboard
2. **Key property**: Black elements have all red neighbors and vice versa
3. **Parallel execution**:
   - Update all red cells in parallel (they don't depend on each other)
   - Then update all black cells in parallel (they don't depend on each other)
4. **Result**: Near Gauss-Seidel convergence speed with near-Jacobi parallelism

**Implementation pattern:**
```rust
For each iteration:
  1. Parallel update of all RED cells (using black neighbor values)
  2. Synchronization barrier
  3. Parallel update of all BLACK cells (using updated red values)
  4. Synchronization barrier
```

**Performance characteristics:**
- Converges faster than Jacobi (close to regular Gauss-Seidel)
- Nearly as parallelizable as Jacobi (50% parallel efficiency per sub-step)
- GPU implementations achieve significant speedups over CPU

**Practical considerations:**
- Grid indexing must correctly identify red vs black cells: `(i + j + k) % 2`
- Requires two kernel launches per iteration (one for each color)
- Memory access patterns can be optimized by processing colors in separate passes

### Advanced: Multigrid Preconditioned Conjugate Gradient

For large-scale or ill-conditioned problems, modern practice favors **Multigrid Preconditioned Conjugate Gradient (MG-PCG)**:

**Key advantages:**
- **Extremely fast convergence**: Often reduces residual by 1 order of magnitude every 2 iterations
- **Scalability**: Handles very large grids (768³ demonstrated) efficiently
- **GPU-friendly**: Modern implementations achieve >90% of theoretical peak performance

**Research finding (McAdams et al., 2010):**
> PCG with multigrid preconditioner typically reduces the residual by one order of magnitude every 2 iterations at 768³ resolution.

**GPU implementation results:**
- With cache-reuse optimized RB-SOR smoother: **5.9x speedup** over basic PCG
- Number of iterations reduced to **less than 9%** of original PCG method
- Strong scaling: **2.1x speedup** going from 64 to 256 GPUs

### Recommendations by Use Case

| Use Case | Recommended Solver | Rationale |
|----------|-------------------|-----------|
| Real-time GPU (games) | Red-Black Gauss-Seidel | Best balance of convergence and GPU parallelism |
| Simple prototypes | Jacobi | Easiest to implement and debug |
| High-quality offline (VFX) | MG-PCG | Fastest total convergence, handles complex domains |
| CPU-based research code | Gauss-Seidel | Simplest efficient option for serial execution |
| Very large grids (>100M cells) | MG-PCG with GPU | Only approach that scales efficiently |

### Sources
- [Comparing Iterative Pressure Solvers](https://openreview.net/pdf?id=mueJ9ZjrfOF)
- [The Gauss-Seidel and Jacobi Methods](https://erkaman.github.io/posts/jacobi_and_gauss_seidel.html)
- [Parallelizing Gauss-Seidel using Graph Coloring](https://erkaman.github.io/posts/gauss_seidel_graph_coloring.html)
- [Linear Solvers for Stable Fluids: GPU vs CPU](https://www3.nd.edu/~zxu2/acms60212-40212-S12/final_project/Linear_solvers_GPU.pdf)
- [A Parallel Multigrid Poisson Solver for Fluids Simulation](https://www.math.ucdavis.edu/~jteran/papers/MST10.pdf)
- [GPU Acceleration of Multigrid PCG](https://dl.acm.org/doi/10.1145/3432261.3432273)

---

## 2. Particle Collision Detection in PIC/FLIP

### The Tunneling Problem

**Tunneling** occurs when fast-moving particles pass completely through collision boundaries between timesteps without being detected. This is particularly problematic in PIC/FLIP simulations where:
- Particles can gain high velocities from pressure forces or particle advection
- Thin obstacles (like cloth or thin walls) are common
- Explicit time integration can allow large displacements per step

### Industry-Standard Approach: Signed Distance Fields (SDF)

**Why SDFs are preferred in FLIP:**

1. **Speed**: Collision queries are O(1) lookups instead of O(n) triangle tests
2. **Already computed**: SDFs are used elsewhere in the simulator (grid-to-surface conversion, etc.)
3. **Smooth gradients**: Provide good surface normals for collision response
4. **Volume queries**: Easy to check if a particle is inside or outside an obstacle

**Basic SDF collision detection:**
```rust
for particle in particles {
    let sdf_value = sample_sdf(particle.position);

    if sdf_value < 0.0 {
        // Particle is inside obstacle
        let surface_normal = sdf_gradient(particle.position);
        particle.position += surface_normal * (-sdf_value);
    }
}
```

### Preventing Tunneling: Swept Collision Detection

**The fundamental technique**: Don't just check the end position - check the path between start and end positions.

**Implementation pattern:**
```rust
fn detect_collision(p_start: Vec3, p_end: Vec3, sdf: &SDF) -> Option<CollisionInfo> {
    // Quick check: end position
    if sdf.sample(p_end) >= 0.0 {
        return None;  // No collision
    }

    // Already stuck
    if sdf.sample(p_start) < 0.0 {
        return Some(handle_stuck_particle(p_start, sdf));
    }

    // Sweep along path
    let direction = (p_end - p_start).normalize();
    let distance = (p_end - p_start).length();
    let num_steps = (distance / min_step_size).ceil() as usize;
    let step_size = distance / num_steps as f32;

    for i in 1..=num_steps {
        let t = i as f32 * step_size / distance;
        let p = p_start + t * (p_end - p_start);

        if sdf.sample(p) < 0.0 {
            // Collision detected, find exact impact point
            return Some(find_exact_collision(p_start, p, sdf));
        }
    }

    None
}
```

**Key finding from FLIP Fluids addon research:**
> "If a particle is close to an obstacle's surface, a collision check is run: A particle's path will be incrementally traced from its start location to its end location. If at any point this path intersects the obstacle's volume, a collision will be detected."

### SDF Quality and Collision Response

**Critical insight**: SDF accuracy degrades with penetration depth.

**Problem:**
- SDF data very close to surface is highly accurate
- Deep inside obstacles, SDF becomes less reliable
- Particles that penetrate deeply may get incorrect surface normals

**Solution** (from FLIP Fluids addon):

```rust
fn handle_collision(particle: &mut Particle, sdf: &SDF) -> bool {
    const MAX_ATTEMPTS: usize = 3;

    for attempt in 0..MAX_ATTEMPTS {
        let sdf_value = sdf.sample(particle.position);

        if sdf_value >= 0.0 {
            return true;  // Successfully outside
        }

        // Project to surface
        let normal = sdf.gradient(particle.position).normalize();
        let projection = particle.position + normal * (-sdf_value * 1.1);

        // Randomize starting position on retry
        if attempt > 0 {
            let offset = random_unit_vector() * (particle.radius * 0.1);
            projection += offset;
        }

        particle.position = projection;

        // Apply collision response
        apply_friction_and_bounce(particle, normal);
    }

    // All attempts failed - revert to previous position
    particle.position = particle.prev_position;
    particle.velocity *= 0.5;  // Damp velocity
    false
}
```

**Collision response workflow:**
1. Use SDF gradient to find surface normal
2. Project particle to closest surface point
3. Apply friction and bounce physics
4. If particle still inside after projection:
   - Randomize starting position for next attempt
   - Re-run collision detection
5. If all attempts fail:
   - Revert to position before collision
   - Clamp velocity

### Spatial Acceleration Structures

For particle-particle collisions, use **spatial hashing**:

**Uniform grid / spatial hashing:**
```rust
// Recommended by Matthias Müller
const CELL_SIZE: f32 = 2.0 * PARTICLE_RADIUS;

struct SpatialHash {
    cells: HashMap<IVec3, Vec<usize>>,  // cell -> particle indices
    cell_size: f32,
}

impl SpatialHash {
    fn hash_position(&self, pos: Vec3) -> IVec3 {
        IVec3::new(
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        )
    }

    fn insert(&mut self, particle_id: usize, position: Vec3) {
        let cell = self.hash_position(position);
        self.cells.entry(cell).or_default().push(particle_id);
    }

    fn query_neighbors(&self, position: Vec3, radius: f32) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let cell = self.hash_position(position);

        // Check 27 neighboring cells (including center)
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor_cell = cell + IVec3::new(dx, dy, dz);
                    if let Some(particles) = self.cells.get(&neighbor_cell) {
                        neighbors.extend(particles);
                    }
                }
            }
        }

        neighbors
    }
}
```

**Dense hash table implementation** (Matthias Müller):
- Two arrays: Particle Array + Count Array
- Count Array stores number of particles per bucket
- Acts as index into Particle Array
- Very efficient O(1) insertion and lookup
- Minimal memory overhead

### Stuck Particle Resolution

**PIC/FLIP specific issue**: Particles can get stuck and accelerate from neighbors.

**Strategies:**
1. **Increase PIC ratio**: Pure PIC (ratio = 1.0) has lower velocities but particles still stick
2. **Randomized collision retry**: Change starting position for collision trace
3. **Velocity damping**: Clamp velocities near boundaries
4. **Multiple collision methods**: Fast SDF method for most particles, slow but accurate particle-level method for difficult cases

### Practical Implementation Recommendations

**Must Have:**
- SDF-based collision detection for all static geometry
- Swept collision detection (incremental path tracing) for fast particles
- Fallback to previous position if collision response fails

**Recommended:**
- Spatial hashing for particle-particle collisions
- Randomized retry for stuck particles
- Separate "difficult particle" queue that uses more expensive CCD

**Optional (for high-quality):**
- Full TOI calculation with binary search
- Speculative CCD for extremely fast particles
- Hierarchical spatial hashing for varied particle sizes

### Open Source Implementations to Study

1. **GridFluidSim3D** (rlguy) - [GitHub](https://github.com/rlguy/GridFluidSim3D)
   - Implements Robert Bridson's methods from "Fluid Simulation for Computer Graphics"
   - Includes SDF-based collision detection
   - C++11 implementation

2. **FLIP-PIC-Fluid-Solver** (austinEng) - [GitHub](https://github.com/austinEng/FLIP-PIC-Fluid-Solver)
   - All geometry objects implement multiple collision detection methods
   - Supports point+distance tolerance and point+ray+timestep queries

3. **WebGL-PIC-FLIP-Fluid** (austinEng) - [GitHub](https://github.com/austinEng/WebGL-PIC-FLIP-Fluid)
   - Based on Zhu & Bridson's SIGGRAPH 2005 "Animating Sand as a Fluid"
   - Web-based, easy to experiment with

### Sources
- [FLIP Fluids Collision Handling Improvements](https://flipfluids.com/weekly-development-notes-50-collision-handling-improvements/)
- [Video Game Physics Tutorial - Collision Detection](https://www.toptal.com/game/video-game-physics-part-ii-collision-detection-for-solid-objects)
- [Houdini FLIP Solver Documentation](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html)
- [Spatial Hashing Tutorial](https://www.gorillasun.de/blog/particle-system-optimization-grid-lookup-spatial-hashing/)
- [NVIDIA GPU Gems 3 - Broad-Phase Collision Detection](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)

---

## 3. Velocity Clamping and Sub-Stepping

### The Stability Problem

Fluid simulations can become unstable due to:
- **Excessive velocities** causing particles to move large distances per timestep
- **Violating CFL condition** (Courant-Friedrichs-Lewy) leading to numerical instability
- **Stiff systems** where different components evolve at vastly different timescales
- **Feedback loops** where high velocities generate even higher velocities

### Jos Stam's Stable Fluids Method (Foundation)

The seminal approach that made real-time fluid simulation practical:

**Key properties:**
- **Unconditionally stable**: Won't explode regardless of timestep size
- **Simple to implement**: Requires only particle tracer and linear interpolator
- **Core techniques:**
  1. Semi-Lagrangian advection for velocity
  2. Implicit sparse linear solve for diffusion
  3. Projection step for divergence-free velocity field

**Why it's stable:**
- Semi-Lagrangian advection can't amplify velocities (diffusive by nature)
- Implicit methods for diffusion are unconditionally stable
- Projection ensures mass conservation

**Trade-off**: Some numerical dissipation (energy loss), but guaranteed stability.

### The CFL Condition

**Definition**: The Courant-Friedrichs-Lewy condition states:
```
CFL = (velocity * timestep) / grid_spacing

For stability: CFL ≤ CFL_max
Where CFL_max depends on the numerical method (typically 0.5 to 1.0)
```

**Physical meaning**: Information can't travel more than one grid cell per timestep.

**Practical interpretation:**
> "The full numerical domain of dependence must contain the physical domain of dependence."

**For explicit methods:**
- CFL > 1.0: Simulation will likely blow up
- CFL = 0.5-1.0: Usually safe
- CFL < 0.5: Conservative, guaranteed stable

**For implicit methods (like Stable Fluids):**
- Can violate CFL and remain stable
- But accuracy degrades with large CFL numbers
- CFL still useful as quality metric

### Velocity Clamping Techniques

**Basic clamping** (last resort):
```rust
let max_velocity = grid_spacing / timestep * max_cfl;
velocity = velocity.clamp_length(0.0, max_velocity);
```

**When to use:**
- Safety net to prevent catastrophic blow-ups
- During initialization or user interaction
- When particles enter simulation from external sources

**Downsides:**
- Non-physical (violates conservation laws)
- Can create visual artifacts (sudden velocity changes)
- Treats symptoms, not root cause

**Better clamping strategy** (from GPU Gems):
- Set texture wrap mode to `CLAMP_TO_EDGE` for velocity fields
- Prevents velocities from being advected outside domain
- Boundary values are clamped, not interior values

**Pressure-based velocity limiting:**
- Use pressure projection to enforce incompressibility
- Divergence removal naturally limits velocities
- Multi-resolution pressure solve can accelerate this

### Sub-Stepping (Time Sub-Division)

**The fundamental technique**: Take multiple smaller timesteps instead of one large timestep.

**Basic sub-stepping:**
```rust
fn simulate(dt: f32) {
    let num_substeps = calculate_substeps(dt, max_velocity);
    let substep_dt = dt / num_substeps as f32;

    for _ in 0..num_substeps {
        advect_particles(substep_dt);
        apply_forces(substep_dt);
        solve_pressure(substep_dt);
        update_grid(substep_dt);
    }
}
```

**Adaptive sub-stepping:**
```rust
fn calculate_substeps(dt: f32, max_velocity: f32, grid_spacing: f32) -> u32 {
    let cfl = (max_velocity * dt) / grid_spacing;
    let target_cfl = 0.5;

    if cfl <= target_cfl {
        1
    } else {
        ((cfl / target_cfl).ceil() as u32).max(1).min(10)  // Cap at 10 substeps
    }
}
```

**When to sub-step:**
- **Collisions**: Always use sub-stepping for collision detection
- **High velocities**: When CFL > 1.0
- **Stiff forces**: Surface tension, viscosity at high Reynolds numbers
- **Particle advection**: FLIP particles often need more substeps than grid

**Performance consideration:**
> "FLIP fluids are faster than SPH fluids, if you don't need to substep the FLIP fluid."

Avoid over-substepping - each substep costs performance.

### Adaptive Time Stepping

**The CFL-based approach**: Instead of fixed timestep, maintain constant CFL number.

**Algorithm:**
```rust
fn adaptive_timestep(
    target_cfl: f32,
    max_velocity: f32,
    grid_spacing: f32,
    min_dt: f32,
    max_dt: f32
) -> f32 {
    if max_velocity < 1e-6 {
        return max_dt;
    }

    let dt = (target_cfl * grid_spacing) / max_velocity;
    dt.clamp(min_dt, max_dt)
}
```

**Advantages:**
- Always uses largest safe timestep
- Reaches convergence in fewest steps
- Automatically slows down when necessary
- Speeds up when fluid is moving slowly

**Key finding from research:**
> "The effectiveness of adaptive time-stepping has been demonstrated to accelerate convergence by 500-1000% compared to CFL-based time-stepping with fixed values."

**AdaptiveCFL ramping** (advanced):
- Adjusts CFL number based on residual convergence
- Doesn't require prior knowledge of simulation complexity
- Can aggressively increase CFL when solver is converging well
- Backs off CFL when convergence stalls

### Position-Based Dynamics (PBD) Approach

An alternative to force-based simulation:

**Key idea**: Directly manipulate positions to satisfy constraints, rather than computing forces.

**Advantages for real-time:**
- Allows large timesteps (0.016s for 60fps games)
- Unconditionally stable
- Simpler to implement than implicit integration

**Application to fluids (Position-Based Fluids):**
- Enforce incompressibility as position constraint
- Artificial pressure term improves particle distribution
- Creates surface tension naturally
- Lower neighbor search requirements than SPH

**Trade-off**: Less physically accurate, but very stable and fast.

### Practical Recommendations

**Must Have (for stability):**
1. Enforce CFL ≤ 1.0 through adaptive timestep or sub-stepping
2. Clamp velocities as safety net (max = grid_spacing / dt)
3. Use implicit methods for diffusion (heat equation)
4. Projection step for incompressibility

**Recommended (for quality):**
1. Adaptive timestep based on max velocity
2. Sub-stepping for collision detection (separate from fluid substeps)
3. Multi-resolution pressure solve for real-time performance
4. Monitor CFL number as diagnostic tool

**Optional (for high-end):**
1. Separate substep counts for grid vs particles
2. Adaptive CFL ramping based on solver convergence
3. Local time stepping for multi-scale phenomena
4. Position-based approach if stability > accuracy

**Debugging Checklist:**

When simulation becomes unstable:
1. ✓ Check CFL number (print max value each frame)
2. ✓ Verify velocity clamping is working
3. ✓ Ensure pressure solver is converging
4. ✓ Look for NaN/Inf values in velocity field
5. ✓ Check boundary conditions (especially CLAMP_TO_EDGE)
6. ✓ Validate timestep isn't too large
7. ✓ Confirm forces are scaled correctly by timestep

### Sources
- [NVIDIA GPU Gems - Fast Fluid Dynamics Simulation](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)
- [Stable Fluids (Jos Stam)](https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf)
- [CFL Condition: How to Choose Your Timestep Size](https://www.simscale.com/blog/cfl-condition/)
- [CFL Condition: The Key to Reliable CFD](https://www.numberanalytics.com/blog/cfl-condition-reliable-cfd)
- [Position Based Fluids](https://mmacklin.com/pbf_sig_preprint.pdf)

---

## 4. Continuous Collision Detection (CCD)

### What is Continuous Collision Detection?

**Discrete Collision Detection** (standard):
- Check if objects overlap at current timestep
- Only considers positions at discrete points in time
- Fast but misses collisions for fast-moving objects

**Continuous Collision Detection** (CCD):
- Checks entire trajectory between previous and current position
- Computes Time of Impact (TOI) when objects first touch
- Slower but catches all collisions including tunneling

**When you need CCD:**
- Bullets, projectiles, or fast-moving rigid bodies
- Thin obstacles (cloth, paper, thin walls)
- High-velocity particles in PIC/FLIP simulations
- Any time discrete detection shows visible tunneling

### The Time of Impact (TOI) Problem

**Definition**: Find time T ∈ [0,1] when two objects first touch.
- T = 0: Objects already touching at start of frame
- T = 1: Objects don't touch before end of frame
- T = 0.5: First contact at mid-frame

**For particles with position P(t) = P₀ + t·v:**
```rust
// Problem: Find smallest T where distance(P(T), obstacle) = 0

// For SDF-based obstacles:
fn find_toi_sdf(p_start: Vec3, p_end: Vec3, sdf: &SDF) -> Option<f32> {
    // Binary search for exact TOI
    let mut t_min = 0.0;
    let mut t_max = 1.0;

    for _ in 0..20 {  // 20 iterations gives ~0.0001% precision
        let t_mid = (t_min + t_max) * 0.5;
        let p = p_start + t_mid * (p_end - p_start);
        let distance = sdf.sample(p);

        if distance.abs() < 1e-4 {
            return Some(t_mid);  // Found surface
        }

        if distance < 0.0 {
            t_max = t_mid;  // Inside, search earlier time
        } else {
            t_min = t_mid;  // Outside, search later time
        }
    }

    if t_max < 1.0 {
        Some(t_max)  // Approximate TOI
    } else {
        None  // No collision
    }
}
```

### CCD Implementation Approaches

#### 1. Motion Clamping (Simplest)

**Algorithm:**
```rust
fn motion_clamping(particle: &mut Particle, sdf: &SDF, dt: f32) {
    let p_start = particle.position;
    let p_end = p_start + particle.velocity * dt;

    if let Some(toi) = find_toi_sdf(p_start, p_end, sdf) {
        // Move only to point of impact
        particle.position = p_start + toi * particle.velocity * dt;
        // Handle collision in next timestep
    } else {
        // No collision, full move
        particle.position = p_end;
    }
}
```

**Characteristics:**
- ✓ Detects nearly all collisions
- ✓ Simple to implement
- ✗ Not physically accurate (object moves less than it should)
- ✗ Can cause slow-motion effect for fast objects

**Best for**: Bullets and projectiles where collision detection matters more than exact physics.

**Quote from research:**
> "For fast objects like bullets it is more important to detect all collisions [than to be physically accurate]."

#### 2. Speculative CCD (Unity/Game Engines)

**Algorithm:**
```rust
fn speculative_ccd(particle: &mut Particle, obstacles: &[Collider], dt: f32) {
    let p_start = particle.position;
    let p_end = p_start + particle.velocity * dt;

    // Expand AABB to include motion
    let motion_aabb = AABB::from_segment(p_start, p_end).expand(particle.radius);

    // Broad phase: Find potential collisions
    let mut contacts = Vec::new();
    for obstacle in obstacles {
        if motion_aabb.intersects(&obstacle.aabb) {
            if let Some(toi) = obstacle.compute_toi(p_start, p_end, particle.radius) {
                contacts.push((obstacle, toi));
            }
        }
    }

    // Sort contacts by TOI
    contacts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Process all contacts in order
    for (obstacle, toi) in contacts {
        // Add contact constraint to solver
        // Solver will ensure all constraints satisfied simultaneously
    }
}
```

**Characteristics:**
- ✓ Prevents tunneling even at extreme velocities
- ✓ Physically based (conserves momentum)
- ✓ Handles multiple simultaneous collisions
- ✗ More expensive than discrete detection
- ✗ Requires constraint solver

**Key insight:**
> "The algorithm is speculative because it picks all potential contacts during the next physics step."

**Best for**: Rigid body physics engines, game physics where accuracy matters.

#### 3. Conservative Advancement (Deformable Objects)

**For deformable/non-convex geometry:**

**Algorithm:**
```rust
fn conservative_advancement(
    particle: &mut Particle,
    obstacle: &Mesh,
    dt: f32
) -> Option<f32> {
    let mut t = 0.0;
    let velocity = particle.velocity;

    while t < 1.0 {
        let pos = particle.position + t * velocity * dt;

        // Compute distance to obstacle
        let distance = obstacle.distance_to(pos);

        // Compute upper bound on relative velocity
        let v_max = velocity.length();

        // Advance time by safe amount
        let delta_t = if v_max > 1e-6 {
            distance / (v_max * dt)
        } else {
            1.0 - t
        };

        t += delta_t;

        // Check for collision
        if distance < TOLERANCE {
            return Some(t);
        }

        if t >= 1.0 {
            break;
        }
    }

    None
}
```

**Characteristics:**
- ✓ Works for any geometry (convex or non-convex)
- ✓ Provably correct (won't miss collisions)
- ✗ Can be slow (many iterations)
- ✗ Requires good distance query

**Best for**: Offline simulation, cloth, deformable bodies.

### CCD for Particle Systems (Special Considerations)

**Challenge**: Particle systems in Unity, Unreal, etc. don't natively support CCD.

**Quote from Unity forums:**
> "It's not possible to apply 'Collision Detection' to the particle [system directly]."

**Workaround strategies:**

1. **Manual swept sphere tests** (lightweight):
   ```rust
   fn particle_ccd(
       p_old: Vec3,
       p_new: Vec3,
       radius: f32,
       obstacles: &[Collider]
   ) -> Option<CollisionInfo> {
       let ray = Ray {
           origin: p_old,
           direction: (p_new - p_old).normalize()
       };
       let max_distance = (p_new - p_old).length();

       for obstacle in obstacles {
           if let Some(hit) = obstacle.raycast_with_radius(ray, radius, max_distance) {
               return Some(CollisionInfo {
                   toi: hit.distance / max_distance,
                   normal: hit.normal,
                   point: hit.point,
               });
           }
       }
       None
   }
   ```

2. **Spatial hashing + swept tests** (recommended):
   ```rust
   fn particle_ccd_optimized(
       particle: &Particle,
       spatial_hash: &SpatialHash,
       obstacles: &[Collider]
   ) -> Option<CollisionInfo> {
       // Only test against nearby obstacles
       let nearby = spatial_hash.query_obstacles(particle.position, particle.velocity.length());

       for obstacle_id in nearby {
           if let Some(collision) = test_ccd(&obstacles[obstacle_id], particle) {
               return Some(collision);
           }
       }
       None
   }
   ```

### Performance Characteristics

**Computational cost hierarchy** (fastest to slowest):

1. **Discrete detection**: O(n) per particle
2. **Motion clamping CCD**: O(n) + TOI computation (still linear)
3. **Speculative CCD**: O(n log n) for broad phase + O(k) narrow phase
4. **Sweep-based CCD**: O(n²) worst case without broad phase
5. **Conservative advancement**: O(n² × iterations)

**Rule of thumb**:
- Use discrete for most particles
- Use CCD only for fast particles (velocity > grid_spacing / timestep)
- Maintain separate queues for "normal" and "fast" particles

### Practical Recommendations

**Must Have (minimum viable CCD):**
1. Swept collision test for particles with velocity > threshold
2. Binary search for TOI with SDF
3. Clamp to first collision (motion clamping)

**Recommended (production quality):**
1. Spatial hashing for broad phase acceleration
2. Separate fast particle queue
3. Speculative CCD for critical particles (player projectiles)
4. Fallback to discrete detection when CCD fails

**Optional (high-end / research):**
1. Conservative advancement for deformable obstacles
2. Provably correct CCD (ACCD methods from NYU research)
3. Parallel CCD on GPU
4. Hierarchical spatial structures (BVH) for complex scenes

**Performance tuning:**
```rust
const VELOCITY_THRESHOLD: f32 = 2.0 * GRID_SPACING / DT;
const CCD_MAX_ITERATIONS: usize = 20;
const CCD_TOLERANCE: f32 = 0.0001;

fn needs_ccd(velocity: Vec3, grid_spacing: f32, dt: f32) -> bool {
    let max_distance = velocity.length() * dt;
    max_distance > grid_spacing
}
```

### Research Resources

**Academic projects:**
- [NYU Geometric Computing Lab CCD](https://continuous-collision-detection.github.io/): Provably correct algorithms, large-scale benchmarks, code on GitHub
- [Self-CCD for deforming objects](http://gamma.cs.unc.edu/SELFCD/): Handles self-collisions in cloth/deformables
- [Defending CCD against errors (ACM TOG)](https://dl.acm.org/doi/10.1145/2601097.2601114): Robust CCD with floating-point guarantees

### Sources
- [Continuous Collision Detection](https://continuous-collision-detection.github.io/)
- [Unity CCD Manual](https://docs.unity3d.com/2020.1/Documentation/Manual/ContinuousCollisionDetection.html)
- [Physics-Based Simulation - CCD Chapter](https://phys-sim-book.github.io/lec21.4-ccd.html)
- [Defending CCD Against Errors](https://dl.acm.org/doi/10.1145/2601097.2601114)
- [Self-CCD for Deforming Objects](http://gamma.cs.unc.edu/SELFCD/)
- [Scalable CCD](https://continuous-collision-detection.github.io/scalable_ccd/)

---

## 5. Summary Recommendations

### High Priority (Must Implement)

1. **Pressure Solver**: Red-Black Gauss-Seidel for GPU, regular Gauss-Seidel for CPU
2. **Collision Detection**: SDF-based with swept path tracing for fast particles
3. **Stability**: Adaptive timestep based on CFL condition (target CFL = 0.5)
4. **Safety Net**: Velocity clamping at grid_spacing / timestep

### Medium Priority (Recommended)

1. **Advanced Solver**: Consider MG-PCG for grids >50M cells
2. **Spatial Acceleration**: Uniform grid / spatial hashing for particle-particle collisions
3. **Sub-stepping**: Separate substep counts for collision vs fluid update
4. **CCD**: Binary search TOI for particles exceeding velocity threshold

### Low Priority (Quality / Optimization)

1. **Multi-resolution**: Coarse pressure solve for real-time applications
2. **Adaptive CFL**: Ramp CFL based on solver convergence
3. **Advanced CCD**: Conservative advancement for deformable obstacles
4. **Local timestepping**: Only for research / offline rendering

---

## 6. Academic References

### Key Papers and Books

1. **Robert Bridson** - "Fluid Simulation for Computer Graphics" (2nd edition)
   - The definitive textbook for graphics-oriented fluid simulation
   - Covers PIC/FLIP, collision handling, pressure solvers
   - [UBC website](https://www.cs.ubc.ca/~rbridson/)

2. **SIGGRAPH Course Notes**:
   - [2007 Fluid Simulation Course (Bridson)](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf)
   - Covers particle-based methods, SPH, collision detection

3. **Jos Stam** - "Stable Fluids" (SIGGRAPH 1999)
   - Foundation of unconditionally stable methods
   - [PDF](https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf)

4. **Zhu & Bridson** - "Animating Sand as a Fluid" (SIGGRAPH 2005)
   - Influential PIC/FLIP paper

5. **McAdams et al.** - "A Parallel Multigrid Poisson Solver" (2010)
   - State-of-the-art pressure solver for large-scale simulation
   - [PDF](https://www.math.ucdavis.edu/~jteran/papers/MST10.pdf)

### Open Source Codebases

1. **GridFluidSim3D**: Production-quality PIC/FLIP in C++
   - https://github.com/rlguy/GridFluidSim3D

2. **FLIP Fluids for Blender**: Commercial-quality addon (open dev blog)
   - https://flipfluids.com

3. **Houdini Documentation**: Industry-standard tool documentation
   - https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html

---

*Research compiled December 20, 2025 for goldrush-fluid-miner project*
