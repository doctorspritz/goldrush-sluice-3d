# APIC (Affine Particle-In-Cell) Implementation Guide

## Overview

APIC (Affine Particle-In-Cell) is a hybrid Lagrangian-Eulerian simulation method that improves upon traditional PIC by storing locally affine velocity fields on particles instead of just constant velocities. This dramatically reduces numerical dissipation while maintaining stability and conserving both linear and angular momentum.

**Original Paper**: Jiang et al., "The Affine Particle-In-Cell Method", ACM Transactions on Graphics (SIGGRAPH 2015)

## 1. Particle Data Structure

Each APIC particle stores:

```rust
struct Particle {
    position: Vec2,     // x_p
    velocity: Vec2,     // v_p
    mass: f32,          // m_p
    C: Mat2,            // Affine velocity matrix (2x2 in 2D, 3x3 in 3D)
}
```

The **C matrix** is the key difference from FLIP/PIC. It stores the local affine velocity field:
```
v(x) = v_p + C_p * (x - x_p)
```

Where:
- `v_p` is the particle velocity
- `C_p` is the 2x2 (or 3x3 in 3D) affine velocity matrix
- `x` is any position near the particle
- `x_p` is the particle position

## 2. P2G Transfer (Particle-to-Grid)

The P2G transfer accumulates both velocity and affine momentum to grid nodes.

### Equations

For each particle p and grid node i:

```
Mass transfer (unchanged from PIC):
m_i = sum_p ( w_ip * m_p )

Momentum transfer (APIC modification):
(m*v)_i = sum_p ( w_ip * m_p * [v_p + C_p * (x_i - x_p)] )
```

Where:
- `w_ip` is the interpolation weight between particle p and grid node i
- `x_i` is the grid node position
- `x_p` is the particle position
- The term `C_p * (x_i - x_p)` adds the affine velocity contribution

### Pseudocode

```rust
// P2G Transfer
fn particle_to_grid(particles: &[Particle], grid: &mut Grid) {
    // Clear grid
    for node in grid.nodes.iter_mut() {
        node.mass = 0.0;
        node.momentum = Vec2::ZERO;
    }

    // Transfer from particles
    for p in particles.iter() {
        let base_coord = (p.position / grid.dx).floor() as i32;

        // Loop over neighboring grid nodes (3x3 in 2D, 3x3x3 in 3D)
        for i in 0..3 {
            for j in 0..3 {
                let grid_pos = Vec2::new(
                    (base_coord.x + i) as f32,
                    (base_coord.y + j) as f32
                ) * grid.dx;

                // Compute interpolation weight (typically quadratic B-spline)
                let weight = compute_weight(p.position, grid_pos, grid.dx);

                // Distance from particle to grid node
                let dpos = grid_pos - p.position;

                // Affine velocity contribution
                let affine_velocity = p.C * dpos;

                // Transfer mass and momentum
                let node = &mut grid.nodes[i][j];
                node.mass += weight * p.mass;
                node.momentum += weight * p.mass * (p.velocity + affine_velocity);
            }
        }
    }

    // Convert momentum to velocity
    for node in grid.nodes.iter_mut() {
        if node.mass > 0.0 {
            node.velocity = node.momentum / node.mass;
        }
    }
}
```

## 3. G2P Transfer (Grid-to-Particle)

The G2P transfer updates both particle velocities AND reconstructs the C matrix.

### Equations

For each particle p:

```
Velocity update:
v_p^new = sum_i ( w_ip * v_i )

C matrix update:
C_p^new = sum_i ( w_ip * v_i * outer_product(∇w_ip) )
       = (1/h²) * sum_i ( w_ip * v_i ⊗ (x_i - x_p)^T )
```

Where:
- `∇w_ip` is the gradient of the interpolation weight
- `⊗` denotes outer product
- `h` is the grid spacing (dx)

**Key insight**: The C matrix is reconstructed from grid velocities using the outer product of weighted velocities and position gradients.

### Pseudocode

```rust
// G2P Transfer
fn grid_to_particle(particles: &mut [Particle], grid: &Grid) {
    for p in particles.iter_mut() {
        let base_coord = (p.position / grid.dx).floor() as i32;

        // Reset particle velocity and C matrix
        let mut new_velocity = Vec2::ZERO;
        let mut new_C = Mat2::ZERO;

        // Loop over neighboring grid nodes
        for i in 0..3 {
            for j in 0..3 {
                let grid_pos = Vec2::new(
                    (base_coord.x + i) as f32,
                    (base_coord.y + j) as f32
                ) * grid.dx;

                let node = &grid.nodes[i][j];

                // Compute interpolation weight
                let weight = compute_weight(p.position, grid_pos, grid.dx);

                // Distance from particle to grid node (in grid coordinates)
                let dpos = (grid_pos - p.position) / grid.dx;

                // Update velocity
                new_velocity += weight * node.velocity;

                // Update C matrix via outer product
                // C += weight * outer_product(grid_velocity, dpos)
                new_C += weight * outer_product(node.velocity, dpos);
            }
        }

        p.velocity = new_velocity;
        p.C = new_C * 4.0; // Scale factor for quadratic B-spline (factor of 4 for 2D)
    }
}

fn outer_product(a: Vec2, b: Vec2) -> Mat2 {
    Mat2::new(
        a.x * b.x, a.x * b.y,
        a.y * b.x, a.y * b.y
    )
}
```

## 4. MLS-MPM Style Implementation (88-line reference)

From the famous Taichi MLS-MPM implementation, here's the compact version:

```cpp
// Particle structure
struct Particle {
    Vec x, v;    // Position, velocity
    Mat F;       // Deformation gradient (for MPM)
    Mat C;       // Affine velocity matrix (APIC)
    real Jp;     // Volume
    int c;       // Material ID
};

// P2G Transfer
for (auto &p : particles) {
    // Compute stress (MPM-specific, omit for pure fluids)
    auto stress = compute_stress(p);

    // Fused APIC momentum + stress
    auto affine = stress + particle_mass * p.C;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            auto dpos = (Vec(i, j) - fx) * dx;  // Distance to grid node
            auto mass_x_velocity = Vec3(p.v * particle_mass, particle_mass);

            grid[base_x + i][base_y + j] +=
                weight[i].x * weight[j].y * (mass_x_velocity + Vec3(affine * dpos, 0));
        }
    }
}

// G2P Transfer
for (auto &p : particles) {
    p.C = Mat(0);
    p.v = Vec(0);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            auto dpos = Vec(i, j) - fx;
            auto grid_v = Vec(grid[base_x + i][base_y + j]);
            auto weight = w[i].x * w[j].y;

            p.v += weight * grid_v;
            p.C += 4 * inv_dx * outer_product(weight * grid_v, dpos);  // APIC C update
        }
    }
}
```

## 5. Key Differences from FLIP/PIC

| Aspect | PIC | FLIP | APIC |
|--------|-----|------|------|
| **Particle data** | v_p | v_p | v_p + C_p |
| **Stability** | Very stable | Unstable | Stable |
| **Dissipation** | High | Low | Low |
| **Noise** | Smooth | Noisy | Smooth |
| **Angular momentum** | Lost | Partially preserved | Exactly conserved |
| **Vorticity** | Poor | Good | Excellent |
| **Complexity** | Simplest | Simple | Moderate |
| **Memory overhead** | Baseline | Baseline | +4 floats (2D), +9 floats (3D) |

**Key code changes from FLIP:**

1. Add `C` matrix to particle structure (2x2 or 3x3)
2. In P2G: Add `C * (x_i - x_p)` to velocity before transferring
3. In G2P: Reconstruct C using outer product of velocity and position gradient
4. Remove FLIP's "velocity delta" transfer (APIC uses pure PIC-style transfer)

## 6. Interpolation Weights

APIC typically uses **quadratic B-spline** interpolation:

```rust
fn quadratic_bspline(x: f32) -> f32 {
    let x = x.abs();
    if x < 0.5 {
        0.75 - x * x
    } else if x < 1.5 {
        0.5 * (1.5 - x) * (1.5 - x)
    } else {
        0.0
    }
}

fn compute_weight_2d(particle_pos: Vec2, grid_pos: Vec2, dx: f32) -> f32 {
    let rel = (particle_pos - grid_pos) / dx;
    quadratic_bspline(rel.x) * quadratic_bspline(rel.y)
}
```

## 7. Performance Considerations

### Computational Cost

- **Per-iteration cost**: Similar to FLIP (~10-20% more expensive)
  - Extra cost: Computing affine term in P2G, outer product in G2P
  - Memory: +4 floats/particle (2D), +9 floats/particle (3D)

- **Overall simulation time**: Often FASTER than FLIP
  - Reason: Better stability allows larger timesteps and fewer substeps
  - Example benchmark: APIC required 1221 timesteps vs FLIP's 1427 timesteps

### When to Use APIC vs FLIP

**Use APIC for:**
- Small-scale simulations (fountains, splashes)
- High vorticity flows (swirling, rotating fluids)
- Viscous fluids (lava, honey)
- When smooth surfaces are critical
- When angular momentum conservation matters

**Use FLIP for:**
- Large-scale simulations (oceans, rivers)
- High-energy, chaotic flows
- When noisy splashes are desirable
- When maximum performance is critical

## 8. Complete Algorithm

```
1. Initialization:
   - Create particles with position, velocity, C = 0

2. Each timestep:
   a. P2G Transfer:
      - Clear grid
      - For each particle:
        * Transfer mass and momentum with affine contribution
      - Divide momentum by mass to get velocity

   b. Grid operations:
      - Apply forces (gravity, etc.)
      - Enforce boundary conditions
      - Solve pressure projection (incompressibility)
      - Update grid velocities

   c. G2P Transfer:
      - For each particle:
        * Interpolate new velocity from grid
        * Reconstruct C matrix via outer product
        * Update position: x += dt * v

   d. Collision handling (particle level)
```

## 9. Common Implementation Pitfalls

1. **Forgetting the scaling factor**: The C matrix update often needs a factor (4 for quadratic B-spline in 2D, 3 in 3D)
2. **Wrong position in outer product**: It's `outer_product(velocity, dpos)` not `outer_product(dpos, velocity)`
3. **Using dx instead of 1/dx**: The C matrix reconstruction uses inverse grid spacing
4. **Initializing C incorrectly**: Start with C = 0, not identity
5. **MAC grid confusion**: On staggered grids, each velocity component lives at face centers, not node centers

## 10. References

### Original Papers
- **Jiang et al. 2015**: "The Affine Particle-In-Cell Method", SIGGRAPH 2015
  - [ACM Link](https://dl.acm.org/doi/10.1145/2766996)
  - [Disney Animation](https://disneyanimation.com/publications/the-affine-particle-in-cell-method/)

- **Jiang et al. 2017**: "An Angular Momentum Conserving Affine-Particle-In-Cell Method"
  - [PDF](https://math.ucdavis.edu/~jteran/papers/JST17.pdf)

### Code Examples
- **Taichi MLS-MPM**: [github.com/yuanming-hu/taichi_mpm](https://github.com/yuanming-hu/taichi_mpm)
  - See `mls-mpm88-explained.cpp` for annotated APIC implementation

- **APIC2D**: [github.com/nepluno/apic2d](https://github.com/nepluno/apic2d)
  - Educational 2D water simulation with APIC, FLIP, PIC modes

### Tutorials
- **Stanford CS348C Assignment**: [Graphics Course](https://graphics.stanford.edu/courses/cs348c-17-fall/PA3_APIC2017/index.html)
- **APIC Writeup**: [alishelton.github.io/apic-writeup](https://alishelton.github.io/apic-writeup/)
- **Gregory Du's Implementation**: [gregoryd2017.github.io](https://gregoryd2017.github.io/home/projects/fluidsim/fluidsim.html)

### Production Use
- **Blender FLIP Fluids**: [APIC solver documentation](https://flipfluids.com/weekly-development-notes-54-new-apic-solver-in-flip-fluids-1-0-9b/)
- **Houdini FLIP Solver**: Includes APIC transfer mode

## 11. Sluice Box Slurry Configuration

For realistic gold sluice simulation, the water-to-solids ratio is critical. Real mining slurry contains 10-12% solids by volume.

### Current Configuration (main.rs)

```rust
// 90% water, 10% solids (realistic slurry)
spawn_rate = 5;           // 5 water particles/frame
mud_rate = 8;             // 1 mud every 8 frames = 0.125/frame
sand_rate = 3;            // 1 sand every 3 frames = 0.33/frame
magnetite_rate = 15;      // 1 black sand every 15 frames = 0.067/frame
gold_rate = 60;           // 1 gold every 60 frames = 0.017/frame
// Total solids ≈ 0.54/frame → ~10% of total particles
```

### Three Scenarios to Test

| Scenario | Solids % | spawn_rate | Behavior |
|----------|----------|------------|----------|
| Too little slurry | 5-7% | 8-10 | Gold washes out with everything else |
| **Ideal** | 10-12% | 5 | Gold/black sand settles in riffles, sand washes out |
| Too much slurry | 15-20% | 2-3 | Riffles pack up, water flows over, nothing settles |

### Expected Physics

1. **Riffle Vortices**: Each riffle creates a low-pressure vortex behind it
2. **Density Separation**: Heavy particles (gold, magnetite) drop into vortex pockets
3. **Light Particles**: Sand and mud stay in suspension, wash downstream
4. **Sorting Process**:
   - Material drops into first few riffles
   - Water stirs and re-sorts
   - Lighter material washes further down
   - Eventually light material exits the sluice

### UI Display

The simulation now shows real-time solids percentage:
```
W:1200 M:15 S:45 Mag:8 Au:2 | Solids: 5.5%
```
