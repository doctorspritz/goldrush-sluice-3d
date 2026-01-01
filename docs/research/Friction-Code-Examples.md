# Friction Implementation: Code Examples from Research

**Source**: Compiled from academic papers and production implementations
**Purpose**: Reference implementations for boundary friction in particle simulations

---

## 1. Taichi MPM - Production Implementation

### Source: yuanming-hu/taichi_mpm

**Friction projection at boundaries** (C++):

```cpp
// Extract boundary properties from signed distance field
Real phi = levelset.sample(pos, t);
Vector n = levelset.get_spatial_gradient(pos, t);
Vector boundary_velocity = -levelset.get_temporal_derivative(pos, t) * n * delta_x;
Real mu = levelset.levelset0->friction;

// Apply friction projection to grid velocity
Vector v = friction_project(grid_velocity(ind), boundary_velocity, n, mu);

// Update velocity-mass grid (preserve mass, update velocity)
VectorP &v_and_m = get_grid(ind).velocity_and_mass;
v_and_m = VectorP(v, v_and_m[dim]);
```

**Friction encoding convention**:
```cpp
// Friction coefficient encoding:
// mu > 0   : Coulomb friction coefficient (e.g., 0.4)
// mu = -1  : Sticky boundary (complete velocity cancellation)
// mu = -2  : Slip boundary (frictionless)
// mu = -2.4: Slip with friction 0.4
```

---

## 2. Matter MPM - Frictional Boundary Conditions

### Source: larsblatny/matter (C++)

**Boundary friction with velocity-relative approach**:

```cpp
// Compute relative velocity (particle velocity relative to boundary)
Vector v_rel = grid_velocity - boundary_velocity;

// Decompose into normal and tangential components
Real v_n = dot(v_rel, normal);
Vector v_n_vec = v_n * normal;
Vector v_tan = v_rel - v_n_vec;
Real v_tan_mag = length(v_tan);

// Apply Coulomb friction
Real friction_threshold = mu_boundary * abs(v_n);

if (v_tan_mag > friction_threshold) {
    // Kinetic friction: reduce tangential velocity
    v_tan *= (1.0 - friction_threshold / v_tan_mag);
} else {
    // Static friction: zero tangential velocity (stick)
    v_tan = Vector(0, 0, 0);
}

// Prevent penetration (cancel inward normal velocity)
if (v_n < 0.0) {
    v_n = 0.0;
}

// Reconstruct grid velocity
grid_velocity = boundary_velocity + v_n * normal + v_tan;
```

**Material-Induced Boundary Friction (MIBF)**:
```cpp
// Use material's internal friction angle as boundary friction
Real phi_internal = material.friction_angle;  // Drucker-Prager friction angle
Real mu_boundary = tan(phi_internal);         // Convert to Coulomb coefficient

// Apply boundary friction using material property
grid_velocity = apply_coulomb_friction(grid_velocity, boundary_velocity, normal, mu_boundary);
```

---

## 3. Houdini FLIP Solver - Particle Friction

### Source: SideFX Houdini Documentation

**Particle-level friction parameters**:

```cpp
struct ParticleCollisionProperties {
    Real friction;      // Coefficient of friction (0.0 to 1.0)
    Real bounce;        // Coefficient of restitution (0.0 to 1.0)
};

// Apply friction and bounce at particle-obstacle collision
void apply_particle_collision(
    Particle &particle,
    Vector collision_normal,
    Vector collision_point,
    ParticleCollisionProperties props
) {
    // Decompose velocity
    Real v_n = dot(particle.velocity, collision_normal);
    Vector v_normal = v_n * collision_normal;
    Vector v_tangent = particle.velocity - v_normal;

    // Apply bounce (restitution) to normal component
    v_normal *= -props.bounce;

    // Apply friction to tangential component
    Real friction_factor = 1.0 - props.friction;
    v_tangent *= friction_factor;

    // Reconstruct velocity
    particle.velocity = v_normal + v_tangent;
}
```

**Collision detection modes**:
```python
# Option 1: Particle-based (accurate but slow)
collision_method = 'particle'  # Checks each particle individually
friction = 0.7
bounce = 0.3

# Option 2: Volume-based (fast but less accurate)
collision_method = 'volume'    # Pushes particles outside collision objects
friction = 0.5
bounce = 0.2
```

---

## 4. Rapier Physics Engine - Rust Implementation

### Source: dimforge/rapier

**Material property friction**:

```rust
use rapier3d::prelude::*;

// Create collider with friction
let collider = ColliderBuilder::cuboid(1.0, 1.0, 0.5)
    .friction(0.7)           // Coefficient of friction
    .restitution(0.3)        // Coefficient of restitution
    .density(2700.0)         // Aluminum density
    .build();

// Material presets
fn ice_material() -> ColliderBuilder {
    ColliderBuilder::new(shape)
        .friction(0.01)
        .restitution(0.1)
}

fn wood_material() -> ColliderBuilder {
    ColliderBuilder::new(shape)
        .friction(0.5)
        .restitution(0.2)
}

fn rubber_material() -> ColliderBuilder {
    ColliderBuilder::new(shape)
        .friction(1.0)
        .restitution(0.8)
}
```

**Custom friction via physics hooks**:

```rust
struct CustomFriction;

impl PhysicsHooks for CustomFriction {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        // Access contact geometry
        let normal = context.normal;
        let tangent1 = context.tangent1;
        let tangent2 = context.tangent2;

        // Modify friction coefficient based on custom logic
        for solver_contact in &mut context.solver_contacts {
            // Example: Velocity-dependent friction
            let relative_velocity = compute_relative_velocity(context);
            let speed = relative_velocity.length();

            if speed < 1.0 {
                // Static friction at low speeds
                solver_contact.friction = 0.8;
            } else {
                // Kinetic friction at high speeds
                solver_contact.friction = 0.5;
            }
        }
    }
}
```

---

## 5. Simplified MLS-MPM - 88 Lines

### Source: yuanming-hu/taichi_mpm (mls-mpm88-explained.cpp)

**Implicit boundary friction** (no explicit coefficients):

```cpp
// Grid-based boundary conditions
Real boundary = 0.05;
Real x = (Real) i / n;
Real y = (Real) j / n;

// Sticky boundary (sides and top) - zero velocity
if (x < boundary || x > 1 - boundary || y > 1 - boundary) {
    grid_velocity = Vector3(0, 0, 0);
}

// Separating boundary (floor) - prevent downward motion only
if (y < boundary) {
    grid_velocity[1] = std::max(0.0f, grid_velocity[1]);  // y-component (vertical)
}
```

**Key insight**: Friction emerges from:
1. **Material plasticity** (hardening parameter)
2. **Velocity clamping** at boundaries
3. **Grid-based constraints** (no explicit μ coefficient)

---

## 6. GeoTaichi - Advanced Friction Features

### Source: Yihao-Shi/GeoTaichi

**Dirichlet boundary with friction**:

```python
# Boundary condition types
boundary_type = {
    'Fix': 0,      # Fixed (zero velocity)
    'Reflect': 1,  # Reflective (mirror velocity)
    'Friction': 2  # Frictional (damped tangential)
}

# Configure frictional boundary
boundary = {
    'type': 'Friction',
    'friction_coefficient': 0.43,  # Coulomb coefficient
    'velocity': [0, 0, 0],          # Boundary velocity (static)
}

# Velocity projection with friction
def apply_friction_boundary(v_particle, v_boundary, normal, mu):
    v_rel = v_particle - v_boundary
    v_n = np.dot(v_rel, normal)
    v_t = v_rel - v_n * normal
    v_t_mag = np.linalg.norm(v_t)

    # Coulomb friction
    threshold = mu * abs(v_n)
    if v_t_mag > threshold:
        v_t *= (1.0 - threshold / v_t_mag)  # Kinetic
    else:
        v_t = np.zeros(3)  # Static

    # Prevent penetration
    if v_n < 0:
        v_n = 0

    return v_boundary + v_n * normal + v_t
```

**Contact model integration**:
```python
# Hertz-Mindlin contact with friction
contact_model = {
    'type': 'HertzMindlin',
    'normal_stiffness': 1e6,
    'shear_stiffness': 5e5,
    'friction_coefficient': 0.5,
    'rolling_friction': 0.1,  # Additional rolling resistance
}
```

---

## 7. APIC MAC Grid - Friction-Aware Transfer

### Source: Fu et al. 2017 (Stanford CS 348C)

**Particle-to-Grid with boundary awareness**:

```cpp
// P2G transfer with friction at boundaries
for (auto &particle : particles) {
    Vector pos = particle.position;
    Vector vel = particle.velocity;
    Matrix C = particle.affine_velocity;

    // Standard APIC transfer
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            int ni = base_i + di;
            int nj = base_j + dj;

            Real weight = quadratic_bspline(delta);
            Vector offset = grid_pos(ni, nj) - pos;
            Vector momentum = (vel + C * offset) * weight;

            grid_velocity[ni][nj] += momentum;
            grid_mass[ni][nj] += weight;
        }
    }
}

// After P2G: Apply boundary friction to grid
for (int i = 0; i < grid_width; i++) {
    for (int j = 0; j < grid_height; j++) {
        if (is_boundary_cell(i, j)) {
            Vector v = grid_velocity[i][j];
            Vector n = boundary_normal(i, j);

            // Friction projection (Coulomb)
            grid_velocity[i][j] = friction_project(v, Vector(0), n, mu);
        }
    }
}
```

**Grid-to-Particle preserves friction effects**:

```cpp
// G2P transfer (standard APIC)
for (auto &particle : particles) {
    Vector new_vel = Vector(0);
    Matrix new_C = Matrix(0);

    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            Real weight = quadratic_bspline(delta);
            Vector v_grid = grid_velocity[ni][nj];
            Vector offset = grid_pos(ni, nj) - particle.position;

            // Velocity transfer
            new_vel += weight * v_grid;

            // Affine matrix update (preserves friction-damped rotation)
            new_C += weight * outer_product(v_grid, offset) * D_inv;
        }
    }

    particle.velocity = new_vel;
    particle.affine_velocity = new_C;
}
```

---

## 8. DC-APIC - Decomposed Compatible Transfer

### Source: Zeng et al. 2025 (Non-sticky solid-fluid interaction)

**Key innovation**: Separate transfers for solid and fluid

```cpp
// Decomposed transfer prevents sticking
void decomposed_p2g_transfer(
    std::vector<Particle> &particles,
    Grid &grid
) {
    // Step 1: Fluid-only transfer (for pressure solve)
    for (auto &p : particles) {
        if (p.is_fluid()) {
            standard_apic_transfer(p, grid);
        }
    }

    // Step 2: Solid-only transfer (for contact resolution)
    for (auto &p : particles) {
        if (p.is_solid()) {
            // Transfer with friction-aware contact
            solid_contact_transfer(p, grid);
        }
    }
}

// Iterative friction enforcement
void iterative_friction_solver(
    std::vector<Particle> &particles,
    Grid &grid,
    int iterations = 5
) {
    for (int iter = 0; iter < iterations; iter++) {
        // Compute contact forces
        for (auto &p : particles) {
            Vector contact_force = compute_contact_force(p, grid);

            // Apply Coulomb friction
            Real f_n = dot(contact_force, p.contact_normal);
            Vector f_t = contact_force - f_n * p.contact_normal;

            Real mu = p.material.friction_coefficient;
            Real max_friction = mu * abs(f_n);

            if (length(f_t) > max_friction) {
                // Sliding: limit friction force
                f_t *= max_friction / length(f_t);
            }

            p.velocity += (f_n * p.contact_normal + f_t) * dt;
        }

        // Update grid with new velocities
        update_grid_from_particles(particles, grid);
    }
}
```

---

## 9. Drucker-Prager Sand - Material Friction

### Source: Klár et al. 2016 (Sand simulation)

**Material-intrinsic friction** (no boundary coefficients):

```cpp
// Drucker-Prager yield criterion encodes friction
struct DruckerPragerMaterial {
    Real cohesion;           // k (yield stress offset)
    Real friction_angle;     // α (friction angle in radians)
    Real hardening_factor;   // ξ (strain hardening)
};

// Yield function: F = √J₂ + α * I₁ - k
bool is_yielding(Matrix stress, DruckerPragerMaterial mat) {
    Real I1 = trace(stress);                    // First invariant
    Real J2 = deviatoric_invariant(stress);     // Second invariant
    Real alpha = mat.friction_angle;
    Real k = mat.cohesion;

    return sqrt(J2) + alpha * I1 > k;
}

// Stress projection (returns to yield surface)
Matrix project_stress(Matrix stress_trial, DruckerPragerMaterial mat) {
    if (!is_yielding(stress_trial, mat)) {
        return stress_trial;  // Elastic: no projection needed
    }

    // Plastic: project to yield surface
    // This naturally dissipates energy (friction-like behavior)
    Matrix dev = deviatoric(stress_trial);
    Real mag = frobenius_norm(dev);

    // Scale deviatoric stress to yield surface
    Real scale = mat.cohesion / (mag + mat.friction_angle * trace(stress_trial));
    return scale * dev + (trace(stress_trial) / 3.0) * Identity;
}
```

**Key insight**: Friction emerges from material model, not boundary condition!

---

## 10. Ferguson-Church Settling with Hindered Effect

### Source: Your current implementation (particle.rs)

**Shape-aware settling velocity**:

```rust
/// Calculate settling velocity using Ferguson-Church universal equation
///
/// Formula: w = (R * g * D²) / (C₁ * ν + √(0.75 * C₂ * R * g * D³))
pub fn settling_velocity(&self, diameter: f32) -> f32 {
    const WATER_DENSITY: f32 = 1.0;
    const GRAVITY: f32 = 150.0;
    const KINEMATIC_VISCOSITY: f32 = 0.5;
    const C1: f32 = 18.0;  // Stokes constant

    let density = self.density();
    let c2 = self.shape_factor();  // Particle shape (1.0 = sphere, 1.8 = flaky gold)

    // Relative submerged density
    let r = (density - WATER_DENSITY) / WATER_DENSITY;

    // Ferguson-Church equation
    let d = diameter;
    let numerator = r * GRAVITY * d * d;
    let denominator = C1 * KINEMATIC_VISCOSITY
        + (0.75 * c2 * r * GRAVITY * d * d * d).sqrt();

    numerator / denominator
}
```

**Hindered settling (Richardson-Zaki)**:

```rust
/// Richardson-Zaki hindered settling factor
/// concentration: volumetric fraction of solids (0.0 to ~0.6)
/// Returns: multiplier for settling velocity (1.0 = no hindrance)
pub fn hindered_settling_factor(concentration: f32) -> f32 {
    const N: f32 = 4.0;  // Richardson-Zaki exponent for fine particles
    let c = concentration.clamp(0.0, 0.6);
    (1.0 - c).powf(N)
}

// Usage in sediment force calculation:
let base_settling = particle.material.settling_velocity(diameter);
let concentration = neighbor_count_to_concentration(neighbor_count, REST_NEIGHBORS);
let hindered_factor = hindered_settling_factor(concentration);
let effective_settling = base_settling * hindered_factor;
```

---

## Summary Table: Friction Approaches

| Method | Friction Type | When to Use | Code Complexity |
|--------|---------------|-------------|-----------------|
| **Taichi MPM** | Velocity projection | Production MPM | Medium |
| **Matter** | Coulomb boundary | Granular materials | Medium |
| **Houdini FLIP** | Particle collision | Fluid-solid interaction | Low |
| **Rapier** | Contact solver | Rigid body dynamics | High |
| **MLS-MPM 88** | Implicit clamping | Rapid prototyping | Very Low |
| **GeoTaichi** | Multi-model friction | Geophysical simulation | High |
| **APIC MAC** | Grid-based friction | Fluid simulation | Medium |
| **DC-APIC** | Iterative separation | Non-sticky contact | High |
| **Drucker-Prager** | Material yield | Sand/granular | Very High |
| **Ferguson-Church** | Settling drag | Sediment transport | Low |

---

## Implementation Priority for Your Project

**Phase 1** (This week): Add **Option 1** from QuickRef
- Simple tangential damping
- 10 lines of code
- Immediate visual improvement

**Phase 2** (Next week): Upgrade to **Coulomb friction**
- Implement `apply_coulomb_friction()` helper
- More physically accurate
- ~30 lines of code

**Phase 3** (Optional): Add **material-specific friction**
- Friction map in Grid struct
- Per-material friction coefficients
- Enables rich gameplay (riffle placement, etc.)

---

**Last Updated**: 2025-12-21
**Related Docs**:
- Comprehensive research: `APIC-MPM-Friction-Research.md`
- Quick implementation guide: `Friction-Implementation-QuickRef.md`
