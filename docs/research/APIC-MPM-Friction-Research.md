# APIC/MPM Friction Implementation Research

**Date**: 2025-12-21
**Project**: Goldrush Fluid Miner
**Focus**: Boundary condition handling and friction integration for APIC particle simulations

---

## Executive Summary

This document compiles research on friction implementation in APIC (Affine Particle-In-Cell), MPM (Material Point Method), and related particle-grid hybrid methods. The research focuses on how friction coefficients are applied during velocity updates, especially at solid boundaries.

**Key Findings**:
1. Most production MPM solvers use **implicit friction** through velocity projection and boundary constraints
2. Explicit Coulomb friction requires **iterative contact resolution** to satisfy momentum conservation
3. APIC's affine velocity transfer naturally supports **friction-aware momentum updates**
4. Rust physics libraries (Rapier) provide well-tested **contact solver patterns** applicable to particle systems

---

## 1. APIC (Affine Particle-In-Cell) Boundary Conditions

### 1.1 Core APIC Method

APIC represents particle velocities as **locally affine** rather than locally constant:
- Stores velocity matrix `C` per particle capturing local velocity gradients
- Transfers momentum: `momentum_i += w_ip * (v_p + C_p * (x_i - x_p))`
- Preserves both linear and angular momentum across transfers

**Reference**: [Jiang et al. 2015 "The Affine Particle-In-Cell Method"](https://dl.acm.org/doi/10.1145/2766996)

### 1.2 Boundary Handling in APIC

When boundaries are non-periodic, APIC uses **ghost cell extrapolation**:
- Extrapolation happens **before** grid-to-particle transfer
- Extrapolation happens **after** particle-to-grid transfer

**Key Challenge**: Traditional PIC/APIC methods suffer from:
1. Material bodies stick at gaps proportional to grid spacing
2. Particles within same grid cell stick permanently
3. This makes friction and rebound challenging

**Solution (DC-APIC 2025)**: Decomposed Compatible APIC separates solid-fluid interactions:
- Enforces contact at essentially **zero gap**
- Novel iterative scheme allows particles to **naturally separate** after contact
- Enforces friction while **satisfying momentum conservation**

**References**:
- [Affine Particle in Cell Method for MAC Grids](https://www.cs.ucr.edu/~craigs/papers/2019-mac-apic/paper.pdf)
- [DC-APIC: Non-sticky Solid-Fluid Interactions](https://www.sciencedirect.com/science/article/pii/S1524070325000165)
- [Stanford CS 348C APIC Assignment](https://graphics.stanford.edu/courses/cs348c-17-fall/PA3_APIC2017/index.html)

### 1.3 APIC Friction Integration

**Hybrid Material Point Method for Frictional Contact** (2019):
> "A momentum preserving frictional contact algorithm based on affine particle-in-cell grid transfers ensures conservation of both linear and angular momentum with a novel use of the APIC method."

The approach uses:
- APIC grid transfers for momentum preservation
- Iterative contact resolution for friction enforcement
- Penalty method for contact forces at discretized boundaries

This prevents the "fictitious gaps" between bodies in contact while maintaining deformability of contact surfaces.

**Reference**: [A Hybrid Material Point Method for Frictional Contact](https://www.researchgate.net/publication/334752040_A_Hybrid_Material_Point_Method_for_Frictional_Contact_with_Diverse_Materials)

---

## 2. MPM (Material Point Method) Friction Implementations

### 2.1 Overview

MPM naturally handles frictional contact through its hybrid Eulerian-Lagrangian nature:
- Original MPM extended by Bardenhagen et al. to include **frictional contact**
- Enabled simulation of granular flow with contact forbidding interpenetration
- Allows separation, sliding, and rolling with friction

**Reference**: [Material Point Method - Wikipedia](https://en.wikipedia.org/wiki/Material_point_method)

### 2.2 Boundary Friction Implementation

**From Matter (v1) Open-Source MPM Solver**:

```text
Boundary friction parameter µb ≥ 0 where velocity v relative to boundary is computed,
with v* denoting relative velocity (at time n+1) before boundary condition is considered.
```

Key features:
- Supports **no-slip** and **frictional** boundary conditions
- Coulomb friction parameter can be supplied
- **Material-Induced Boundary Friction (MIBF)**: Uses internal friction parameter of plastic model as Coulomb friction for terrain-material interaction
- Supports quadratic and cubic B-splines for particle-grid interpolation

**Code Pattern** (from larsblatny/matter):
```cpp
// Compute relative velocity at boundary
Vector v_rel = grid_velocity - boundary_velocity;
Vector v_tan = v_rel - v_n * n;  // Tangential component
Vector v_n_vec = v_n * n;         // Normal component

// Apply Coulomb friction
real friction_threshold = mu_b * abs(v_n);
if (v_tan.length() > friction_threshold) {
    // Kinetic friction: reduce tangential velocity
    v_tan *= (1.0 - friction_threshold / v_tan.length());
} else {
    // Static friction: zero tangential velocity
    v_tan = Vector(0);
}

// Update velocity
grid_velocity = v_n_vec + v_tan + boundary_velocity;
```

**References**:
- [Matter (v1) MPM Solver for Granular Matter](https://gmd.copernicus.org/articles/18/9149/2025/)
- [larsblatny/matter GitHub](https://github.com/larsblatny/matter/)
- [GMD - Matter (v1) Paper](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1157/egusphere-2025-1157.pdf)

### 2.3 Recent Advances: High-Fidelity Frictional Contact (2024)

**Novel MPM approach** using penalty method for contact forces:
- Evaluates contact forces at **discretized boundaries** of physical domains
- Enhances fidelity by accounting for **deformability of contact surface**
- Prevents fictitious gaps between bodies in contact

Uses **Extended B-Splines (EBSs)**:
- Robustly mitigates grid cell-crossing errors
- Provides continuous gradients of basis functions at cell interfaces
- Minimizes numerical integration errors

**Reference**: [A High-Fidelity Material Point Method for Frictional Contact](https://arxiv.org/html/2403.13534v1)

### 2.4 Coulomb Friction Model in MPM

**Velocity-dependent friction** from soft-rigid contact model:
```text
Coulomb friction model: τ_max = μ * σ_n
where:
  τ_max = maximum shear stress
  μ = friction coefficient
  σ_n = normal stress
```

**Implementation notes**:
- MPM naturally results in **sticking contact** (no interpenetration, no slip) with same shape functions
- Velocities undergo **jumps** from collisions and Coulomb friction's non-smooth nature
- Implicit MPM schemes motivated by implicit time-stepping in contact dynamics (CD) methods

**References**:
- [A Soft-Rigid Contact Model of MPM](https://link.springer.com/article/10.1007/s40571-018-0188-5)
- [Contact and Friction Simulation for Computer Graphics (SIGGRAPH 2022)](https://siggraphcontact.github.io/assets/files/SIGGRAPH22_friction_contact_notes.pdf)

---

## 3. PIC/FLIP Boundary Friction Methods

### 3.1 Velocity Update Methods

**PIC vs FLIP approaches**:
- **PIC**: Particles updated with new grid velocity directly
- **FLIP**: Particle velocity incremented by **delta** in grid velocity
- **Hybrid**: Common to use 5-10% PIC + 90-95% FLIP for stability

```cpp
// FLIP velocity update
v_particle = v_particle + (v_grid_new - v_grid_old);

// PIC velocity update
v_particle = v_grid_new;

// Hybrid (90% FLIP, 10% PIC)
v_particle = 0.9 * (v_particle + delta_v_grid) + 0.1 * v_grid_new;
```

**Reference**: [Fluid Simulation Using Implicit Particles](http://www.danenglesson.com/images/portfolio/FLIP/rapport.pdf)

### 3.2 Boundary Conditions

As **Bridson** states:
> "Most of the, ahem, 'fun' in numerically simulating fluids is in getting the boundary conditions correct."

Three boundary types:
1. **Solid walls** (no-slip or free-slip)
2. **Free surfaces** (freely moving)
3. **Other liquids** (interface conditions)

**Particle collision methods** (from Houdini FLIP solver):
- **Particle detection**: Most accurate but slowest, supports friction and bounce
- **Volume-based**: Faster, moves particles outside collision objects
- **Parameters**: Friction (coefficient) and Bounce (restitution)

**Reference**: [Houdini FLIP Solver Documentation](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html)

### 3.3 Cell Type Marking

```cpp
// Mark cells based on contents
enum CellType { SOLID, FLUID, AIR };

// Important for pressure equations setup
if (particle_in_cell(i, j)) {
    cell_type[i][j] = FLUID;
}
if (terrain_solid(i, j)) {
    cell_type[i][j] = SOLID;
}
```

This classification drives pressure solve boundary conditions.

**Reference**: [Ten Minute Physics FLIP Tutorial](https://matthias-research.github.io/pages/tenMinutePhysics/18-flip.pdf)

---

## 4. Rust Physics Libraries (Rapier/Parry)

### 4.1 Rapier Friction Implementation

**Material presets** showing friction coefficients:

```rust
// Ice
.friction(0.01).restitution(0.1)

// Wood
.friction(0.5).restitution(0.2)

// Rubber
.friction(1.0).restitution(0.8)

// Metal
.friction(0.6).restitution(0.3)
```

**Friction in contact solving**:
```rust
let collider = ColliderBuilder::cuboid(1.0, 1.0)
    .friction(0.7)
    .restitution(0.3)
    .build();
```

**Reference**: [Rapier Documentation](https://context7.com/dimforge/rapier/llms.txt)

### 4.2 Contact Force Events

Rapier provides **contact force callbacks** for custom friction models:

```rust
// Process contact force events
while let Ok(contact_event) = contact_force_recv.try_recv() {
    println!("Contact force magnitude: {}", contact_event.total_force_magnitude);
    println!("Max force direction: {:?}", contact_event.max_force_direction);
}
```

This allows implementing **custom friction** based on:
- Normal force magnitude
- Tangential velocity
- Material properties

### 4.3 Physics Hooks for Custom Friction

**Modify solver contacts** using hooks:

```rust
impl PhysicsHooks for CustomFriction {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        // Access normal and tangential components
        let normal = context.normal;
        let tangent1 = context.tangent1;
        let tangent2 = context.tangent2;

        // Modify friction coefficient based on custom logic
        context.solver_contacts[0].friction = custom_friction_coefficient();
    }
}
```

**Reference**: [Rapier GitHub Claude.md](https://github.com/dimforge/rapier/blob/master/Claude.md)

---

## 5. Disney/Pixar Research Papers

### 5.1 Snow Simulation (Stomakhin et al. 2013)

**"A Material Point Method for Snow Simulation"** - Used in Frozen

Key contributions:
- User-controllable **elasto-plastic constitutive model**
- Integrated with hybrid Eulerian/Lagrangian MPM
- Snow treated as elasto-plastic material:
  - Elastic behavior below stress threshold
  - Plastic deformation above threshold

**Friction handling**: Implicit through material model, not explicit boundary friction

**Impact**: Used in ~43 scenes in Frozen, also in Big Hero 6 and Zootopia

**References**:
- [Alexey Stomakhin's Research Page](https://alexey.stomakhin.com/research/snow.html)
- [Walt Disney Animation Studios Publication](https://disneyanimation.com/publications/a-material-point-method-for-snow-simulation/)
- [ACM Digital Library](https://dl.acm.org/doi/10.1145/2461912.2461948)

### 5.2 Sand Simulation (Klár et al. 2016)

**"Drucker-Prager Elastoplasticity for Sand Animation"**

Key innovation:
- **Drucker-Prager plastic flow model** with Hencky-strain hyperelasticity
- Naturally represents **frictional relation** between shear and normal stresses
- Yield stress criterion enforced via **stress projection algorithm**
- Works with both implicit and explicit time integration

**Friction model**:
```text
Drucker-Prager yield criterion:
  √(J₂) + α * I₁ ≤ k

where:
  J₂ = second invariant of deviatoric stress
  I₁ = first invariant of stress tensor
  α = friction angle parameter
  k = cohesion
```

The friction angle `α` naturally relates shear stress to normal stress, providing **material-intrinsic friction** without separate boundary friction parameters.

**References**:
- [Drucker-Prager Elastoplasticity for Sand Animation (PDF)](https://math.ucdavis.edu/~jteran/papers/KGPSJT16.pdf)
- [ACM Digital Library](https://dl.acm.org/doi/10.1145/2897824.2925906)

### 5.3 General MPM Course (Stomakhin et al. 2016)

**"The Material Point Method for Simulating Continuum Materials"** - SIGGRAPH 2016 Course

Comprehensive overview of MPM for:
- Elastic objects
- Snow
- Lava
- Sand
- Viscoelastic fluids

**Key insight for friction**: MPM's strength is in **material constitutive models** rather than explicit boundary friction. Energy dissipation comes from plasticity, viscosity, and material behavior.

**Reference**: [SIGGRAPH 2016 MPM Course (PDF)](https://alexey.stomakhin.com/research/siggraph2016_mpm.pdf)

---

## 6. Taichi MPM Implementation Examples

### 6.1 Friction Encoding Convention (yuanming-hu/taichi_mpm)

Taichi uses **numerical encoding** for friction types:

```python
# Friction coefficient encoding:
# Positive values: coefficient of friction (e.g., 0.4)
# -1: Sticky (complete velocity cancellation)
# -2: Slip (frictionless)
# -2.4: Slip with friction 0.4
```

**Code pattern** (from Taichi Elements):
```python
surface_sticky = 0  # Stick to boundary
surface_slip = 1    # Slippy boundary
surface_separate = 2  # Slippy and free to separate

# Object friction parameters
restitution = 0.0  # Coefficient of restitution
friction = 0.0     # Coefficient of friction
```

**References**:
- [yuanming-hu/taichi_mpm GitHub](https://github.com/yuanming-hu/taichi_mpm)
- [taichi_elements GitHub](https://github.com/taichi-dev/taichi_elements)

### 6.2 Boundary Friction Velocity Update

**From taichi_mpm/src/mpm.cpp**:

```cpp
// Extract boundary properties from levelset
phi = levelset.sample(pos, t);
n = levelset.get_spatial_gradient(pos, t);
boundary_velocity = -levelset.get_temporal_derivative(pos, t) * n * delta_x;
mu = levelset.levelset0->friction;

// Apply friction projection
Vector v = friction_project(grid_velocity(ind), boundary_velocity, n, mu);

// Update velocity-mass grid
VectorP &v_and_m = get_grid(ind).velocity_and_mass;
v_and_m = VectorP(v, v_and_m[dim]);  // Preserve mass, update velocity
```

**Friction projection function** (inferred pattern):
```cpp
Vector friction_project(Vector v_grid, Vector v_boundary, Vector n, real mu) {
    // Compute relative velocity
    Vector v_rel = v_grid - v_boundary;

    // Decompose into normal and tangential
    real v_n = dot(v_rel, n);
    Vector v_t = v_rel - v_n * n;

    // Apply Coulomb friction
    if (mu < 0) {
        // Special case: sticky boundary
        return v_boundary;
    } else if (mu == 0) {
        // Frictionless: preserve tangential, cancel normal penetration
        if (v_n < 0) v_n = 0;
        return v_boundary + v_n * n + v_t;
    } else {
        // Coulomb friction
        real friction_force = mu * abs(v_n);
        real v_t_mag = v_t.length();

        if (v_t_mag > friction_force) {
            // Kinetic friction
            v_t *= (1.0 - friction_force / v_t_mag);
        } else {
            // Static friction: stick
            v_t = Vector(0);
        }

        if (v_n < 0) v_n = 0;
        return v_boundary + v_n * n + v_t;
    }
}
```

**Reference**: [taichi_mpm/src/mpm.cpp](https://github.com/yuanming-hu/taichi_mpm/blob/master/src/mpm.cpp)

### 6.3 Simple MLS-MPM (88 lines)

The **minimalist MLS-MPM** implementation shows **implicit boundary friction**:

```cpp
real boundary = 0.05;
real x = (real) i / n;
real y = real(j) / n;

// Sticky boundary (sides and top)
if (x < boundary || x > 1-boundary || y > 1-boundary) {
    g = Vector3(0);  // Zero velocity at sticky walls
}

// Separating boundary (floor)
if (y < boundary) {
    g[1] = std::max(0.0f, g[1]);  // Prevent downward penetration
}
```

**Key observation**: No explicit friction coefficients. Friction-like behavior emerges from:
1. Material plasticity and hardening
2. Geometric boundary constraints
3. Velocity clamping at boundaries

**Reference**: [mls-mpm88-explained.cpp](https://github.com/yuanming-hu/taichi_mpm/blob/master/mls-mpm88-explained.cpp)

### 6.4 GeoTaichi Advanced Features

**GeoTaichi** provides production-ready friction:
- Dirichlet boundary conditions: Fix/Reflect/**Friction**
- Contact models: Linear elastic, Hertz-Mindlin, Energy conserving (Barrier functions)
- Supports **DEM-MPM-Mesh contact** for complex boundaries
- TPIC/APIC/MLS velocity projection techniques

**Reference**: [GeoTaichi GitHub](https://github.com/Yihao-Shi/GeoTaichi)

---

## 7. Implementation Recommendations for Your Project

### 7.1 Current Implementation Analysis

Your `/Users/simonheikkila/Documents/antigravity-dev/goldrush-fluid-miner/crates/sim/src/flip.rs` already has:

✅ **APIC particle-to-grid transfer** with affine velocity matrix
✅ **SDF-based boundary collision** with gradient-based position correction
✅ **Velocity reflection** at solid boundaries (removes normal component)

**Current boundary handling** (lines 905-923):
```rust
// SDF collision: sample distance to nearest solid
let sdf_dist = grid.sample_sdf(particle.position);

if sdf_dist < cell_size * 0.5 {
    // Get gradient (points away from solid)
    let grad = grid.sdf_gradient(particle.position);

    // Push out to safe distance
    let push_dist = cell_size * 0.5 - sdf_dist;
    particle.position += grad * push_dist;

    // Remove velocity component into solid
    let v_dot_n = particle.velocity.dot(grad);
    if v_dot_n < 0.0 {
        particle.velocity -= grad * v_dot_n;  // ← NO FRICTION HERE
    }
}
```

**Missing**: Tangential velocity friction at boundaries.

### 7.2 Recommended Friction Integration

**Option 1: Simple Tangential Damping** (Easy, visually effective)

```rust
// In advect_particles(), replace velocity reflection with friction
if sdf_dist < cell_size * 0.5 {
    let grad = grid.sdf_gradient(particle.position);
    let push_dist = cell_size * 0.5 - sdf_dist;
    particle.position += grad * push_dist;

    // Decompose velocity into normal and tangential
    let v_n = particle.velocity.dot(grad);
    let v_normal = grad * v_n;
    let v_tangent = particle.velocity - v_normal;

    // Apply friction to tangential component
    let friction_coeff = 0.3; // Wall friction (0 = frictionless, 1 = maximum)
    let v_tangent_damped = v_tangent * (1.0 - friction_coeff);

    // Cancel inward normal velocity
    let v_normal_clamped = if v_n < 0.0 { Vec2::ZERO } else { v_normal };

    particle.velocity = v_tangent_damped + v_normal_clamped;
}
```

**Option 2: Coulomb Friction** (Physically accurate, more complex)

```rust
fn apply_boundary_friction(
    particle_vel: Vec2,
    boundary_normal: Vec2,
    boundary_vel: Vec2,
    friction_coeff: f32,
) -> Vec2 {
    // Relative velocity
    let v_rel = particle_vel - boundary_vel;

    // Decompose
    let v_n = v_rel.dot(boundary_normal);
    let v_n_vec = boundary_normal * v_n;
    let v_t = v_rel - v_n_vec;
    let v_t_mag = v_t.length();

    // Coulomb friction threshold
    let friction_threshold = friction_coeff * v_n.abs();

    let v_t_new = if v_t_mag > friction_threshold {
        // Kinetic friction: reduce tangential velocity
        v_t * (1.0 - friction_threshold / v_t_mag)
    } else {
        // Static friction: zero tangential velocity
        Vec2::ZERO
    };

    // Prevent penetration
    let v_n_new = if v_n < 0.0 { 0.0 } else { v_n };

    boundary_vel + boundary_normal * v_n_new + v_t_new
}
```

**Option 3: Material-Specific Friction** (Most realistic)

Add friction coefficient to terrain SDF:

```rust
pub struct Grid {
    // ... existing fields ...
    friction_map: Vec<f32>,  // Per-cell friction coefficient
}

impl Grid {
    pub fn sample_friction(&self, pos: Vec2) -> f32 {
        // Bilinear interpolation of friction values
        // Returns friction coefficient at position
        // Examples: 0.1 for ice, 0.3 for rock, 0.7 for wood
    }
}

// In particle advection:
let friction = grid.sample_friction(particle.position);
particle.velocity = apply_boundary_friction(
    particle.velocity,
    grad,
    Vec2::ZERO,  // Static boundary
    friction,
);
```

### 7.3 Sediment-Specific Considerations

For **gold panning simulation**, friction is critical for:

1. **Sluice riffles**: High friction (0.5-0.8) to trap heavy particles
2. **Smooth chutes**: Low friction (0.1-0.3) for light sediment transport
3. **Textured mats**: Variable friction for separation

**Recommendation**: Start with **Option 1** for immediate visual improvement, then add **Option 3** for gameplay-relevant material interactions.

### 7.4 Integration Points in Your Code

**File**: `/Users/simonheikkila/Documents/antigravity-dev/goldrush-fluid-miner/crates/sim/src/flip.rs`

**Location 1** - Particle advection (line 905-923):
```rust
fn advect_particles(&mut self, dt: f32) {
    // ... existing code ...

    // ADD FRICTION HERE after SDF collision
    if sdf_dist < cell_size * 0.5 {
        let grad = grid.sdf_gradient(particle.position);

        // [INSERT FRICTION CODE HERE]
        particle.velocity = apply_boundary_friction(
            particle.velocity,
            grad,
            Vec2::ZERO,
            0.3  // Wall friction coefficient
        );
    }
}
```

**Location 2** - Grid boundary conditions (line 113):
```rust
// After enforce_boundary_conditions(), add friction to grid velocities
self.grid.enforce_boundary_conditions();
self.grid.apply_boundary_friction(0.3);  // New method
```

**Location 3** - Material properties:
```rust
// In particle.rs, add friction property
impl ParticleMaterial {
    pub fn boundary_friction(&self) -> f32 {
        match self {
            Self::Water => 0.0,       // Frictionless
            Self::Mud => 0.4,         // Moderate
            Self::Sand => 0.3,        // Low
            Self::Magnetite => 0.35,  // Medium
            Self::Gold => 0.5,        // Higher (catches on roughness)
        }
    }
}
```

---

## 8. Testing and Validation

### 8.1 Visual Tests

**Test 1**: Particle sliding down inclined plane
- Expected: Without friction, particles accelerate indefinitely
- With friction: Terminal velocity reached, depends on friction coefficient

**Test 2**: Gold particles in sluice riffles
- Expected: High friction at riffles traps gold
- Low friction areas allow light sediment to wash away

**Test 3**: Particle stacking against walls
- Expected: Particles should stack without sliding through walls
- Tangential friction prevents unrealistic "wall climbing"

### 8.2 Quantitative Validation

**Conservation checks**:
```rust
// Before and after friction application
let momentum_before = particles.iter().map(|p| p.velocity).sum();
let momentum_after = particles.iter().map(|p| p.velocity).sum();

// Friction should dissipate energy, not create it
assert!(energy_after <= energy_before);

// Momentum should be conserved for interior particles
// (only boundary interactions should change total momentum)
```

### 8.3 Performance Considerations

- **Simple damping** (Option 1): Negligible overhead (~2-3 FLOPs per boundary particle)
- **Coulomb friction** (Option 2): ~10-15 FLOPs per boundary particle
- **Material-specific** (Option 3): +1 texture lookup per boundary particle

For your simulation with ~10k particles and ~5% boundary interactions:
- Impact: <1ms additional per frame at 60 FPS
- Recommendation: **Implement Option 1 first**, profile, then enhance if needed

---

## 9. References and Further Reading

### Academic Papers

1. **Jiang et al. 2015**: "The Affine Particle-In-Cell Method" - [ACM Digital Library](https://dl.acm.org/doi/10.1145/2766996)
2. **Stomakhin et al. 2013**: "A Material Point Method for Snow Simulation" - [Disney Animation](https://disneyanimation.com/publications/a-material-point-method-for-snow-simulation/)
3. **Klár et al. 2016**: "Drucker-Prager Elastoplasticity for Sand Animation" - [ACM](https://dl.acm.org/doi/10.1145/2897824.2925906)
4. **Fu et al. 2017**: "A Polynomial Particle-In-Cell Method" - [ACM](https://dl.acm.org/doi/10.1145/3130800.3130878)
5. **Zeng et al. 2025**: "DC-APIC: Decomposed Compatible APIC" - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1524070325000165)

### Open-Source Implementations

6. **yuanming-hu/taichi_mpm**: High-performance MLS-MPM - [GitHub](https://github.com/yuanming-hu/taichi_mpm)
7. **larsblatny/matter**: MPM with frictional boundaries - [GitHub](https://github.com/larsblatny/matter/)
8. **dimforge/rapier**: Rust physics engine with friction - [GitHub](https://github.com/dimforge/rapier)
9. **GeoTaichi**: Production MPM solver - [GitHub](https://github.com/Yihao-Shi/GeoTaichi)

### Textbooks and Courses

10. **Bridson 2015**: "Fluid Simulation for Computer Graphics" - [Book](https://www.cs.ubc.ca/~rbridson/fluidsimulation/)
11. **Stomakhin et al. 2016**: "The Material Point Method" SIGGRAPH Course - [PDF](https://alexey.stomakhin.com/research/siggraph2016_mpm.pdf)
12. **SIGGRAPH 2022**: "Contact and Friction Simulation" Course - [Notes](https://siggraphcontact.github.io/assets/files/SIGGRAPH22_friction_contact_notes.pdf)

### Online Resources

13. **Ten Minute Physics**: FLIP Water Simulation - [Tutorial](https://matthias-research.github.io/pages/tenMinutePhysics/18-flip.pdf)
14. **Stanford CS 348C**: APIC Assignment - [Course](https://graphics.stanford.edu/courses/cs348c-17-fall/PA3_APIC2017/index.html)
15. **fxguide**: The Science of Fluid Sims - [Article](https://www.fxguide.com/fxfeatured/the-science-of-fluid-sims/)

---

## 10. Conclusion

### Key Takeaways

1. **APIC naturally supports friction** through affine velocity preservation and momentum-conserving transfers
2. **Most production solvers use implicit friction** via boundary constraints and material models rather than explicit Coulomb friction
3. **Explicit friction requires careful implementation** to maintain momentum conservation and prevent numerical artifacts
4. **Simple tangential damping** (friction coefficient 0-1) provides 80% of visual benefit with 20% of implementation complexity
5. **Material-specific friction** enables rich gameplay for gold panning simulation

### Next Steps for Your Project

**Immediate** (1-2 hours):
- [ ] Implement Option 1 (simple tangential damping) in `advect_particles()`
- [ ] Add friction coefficient parameter to `Grid` struct
- [ ] Test with riffles in sluice box

**Short-term** (1 week):
- [ ] Add material-specific friction coefficients to `ParticleMaterial`
- [ ] Implement `sample_friction()` for spatial friction variation
- [ ] Profile performance impact on 10k+ particle simulations

**Long-term** (1 month):
- [ ] Consider Coulomb friction for physics accuracy if needed
- [ ] Add friction parameter tuning to UI
- [ ] Validate against real-world gold panning behavior

**Optional Enhancement**:
- [ ] Study Rapier's contact solver for particle-particle friction
- [ ] Implement DC-APIC for improved solid-fluid separation
- [ ] Add friction heat dissipation for thermodynamic accuracy (probably overkill!)

---

**Document Version**: 1.0
**Author**: Claude Opus 4.5 (Research Assistant)
**Last Updated**: 2025-12-21
