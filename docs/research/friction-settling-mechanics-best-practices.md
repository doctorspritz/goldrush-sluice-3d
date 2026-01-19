# Friction and Settling Mechanics in APIC/FLIP Simulations: Best Practices

**Research Date:** 2025-12-21
**Focus:** Practical implementation approaches for granular/sediment particles in real-time fluid simulations

## Executive Summary

This document synthesizes best practices for implementing friction and settling mechanics in APIC/FLIP fluid simulations, with emphasis on granular and sediment particle behavior. The research covers six key areas: Coulomb friction models, Shields criterion for sediment transport, APIC boundary friction, granular material physics, sluice/flume sediment dynamics, and Stokes settling velocity.

---

## 1. Coulomb Friction Models

### Overview

The Coulomb friction model distinguishes between **static** and **dynamic friction** for particles on surfaces. Two distinct responses are possible: **stick** and **slip**, with the transition between these states being a common source of algorithmic instability.

### Key Concepts

**Static vs Dynamic Friction:**
- **Static friction** prevents motion until critical shear stress is exceeded
- **Dynamic friction** applies during sliding motion
- Typical relationship: μ_dynamic ≈ 0.7-0.9 × μ_static

**Basic Model:**
```
F_friction = μ × N
```
where:
- F_friction = friction force
- μ = coefficient of friction
- N = normal force

### Implementation Approaches

#### 1. Direct Coulomb Implementation
- **Pros:** Physically accurate, straightforward
- **Cons:** Discontinuous at zero velocity, can cause numerical instability

#### 2. Regularized Friction
- Use continuous function to approximate discontinuity at zero velocity
- **Example:** Tanh smoothing: `F = μ × N × tanh(v/v_threshold)`
- Trade-off: Small slope parameter needed to avoid "creep" at low velocities

#### 3. Linear Complementarity Problem (LCP) Formulation
- [Tonge et al.](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14727) formulated frictional constraints using boxed LCPs
- Pyramid approximation to Coulomb friction cone
- Projected Gauss-Seidel (PGS) method for solving without approximation

#### 4. SPH/Particle Method Implementation
- [Probst et al. 2023](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14727) proposed explicit Coulomb friction for rigid body-fluid coupling
- **Limitation:** Cannot reproduce static friction perfectly
- Not guaranteed to produce physically correct friction magnitude

### Discrete Element Method (DEM) Friction

For 3D particle simulations, [numerically exact Coulomb friction relations](https://www.epj-conferences.org/articles/epjconf/abs/2021/03/epjconf_pg2021_14005/epjconf_pg2021_14005.html) have been developed extending 2D work to three dimensions.

**Key challenges:**
- Handling stick-slip transitions
- Preventing creep in quasi-static scenarios
- Maintaining stability during contact resolution

### Practical Recommendations

1. **For Real-Time Applications:**
   - Use regularized friction with carefully tuned threshold velocity
   - Implement velocity-dependent friction: `μ(v) = μ_s - (μ_s - μ_d) × (1 - e^(-v/v_c))`

2. **For High-Fidelity Simulations:**
   - Implement proper LCP formulations with cone constraints
   - Use iterative solvers (PGS, Lemke) for exact friction cones

3. **Stability Tips:**
   - Avoid zero-crossing discontinuities
   - Use implicit integration for friction forces when possible
   - Implement friction force limits based on contact normal impulse

**Sources:**
- [Implementation of numerically exact Coulomb friction for DEM](https://www.epj-conferences.org/articles/epjconf/abs/2021/03/epjconf_pg2021_14005/epjconf_pg2021_14005.html)
- [Monolithic Friction and Contact Handling for Rigid Bodies and Fluids](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14727)
- [20-sim Friction Modeling Tutorial](https://www.20sim.com/webhelp/modeling_tutorial_friction_staticdynamicphenomena.php)

---

## 2. Shields Criterion for Sediment Transport

### Overview

The [Shields parameter](https://en.wikipedia.org/wiki/Shields_parameter) (θ or τ*) is a dimensionless number used to determine the **initiation of motion** of sediment particles in fluid flow. Developed by Albert F. Shields in 1936, it remains the foundational criterion for sediment transport.

### Mathematical Definition

```
θ = τ / ((ρ_s - ρ_f) × g × d)
```

where:
- τ = bed shear stress
- ρ_s = sediment density
- ρ_f = fluid density
- g = gravitational acceleration
- d = particle diameter

**Physical Interpretation:**
The Shields parameter is proportional to the ratio of **fluid force on particle** to **weight of particle**.

### Critical Shields Stress (θ_c)

**Typical value:** θ_c ≈ 0.03-0.06 for most sediments

Movement occurs when: θ > θ_c

**Factors affecting θ_c:**
- Particle Reynolds number (Re_p)
- Grain shape, orientation, and protrusion
- Bed slope
- Particle size distribution (hiding/exposure effects)
- Vertical particle motion in accelerating/decelerating flows

### Implementation in Transport Models

#### HEC-RAS Approach
[HEC-RAS](https://www.hec.usace.army.mil/confluence/rasdocs/d2sd/ras2dsedtr/6.6/model-description/critical-thresholds-for-transport-and-erosion) automatically selects critical Shields formulation based on transport equation:

1. **Constant Shields parameter** (simple approach)
2. **Function of dimensionless grain diameter** (more sophisticated)

```rust
// Example implementation
fn critical_shields_stress(d_star: f32) -> f32 {
    // Dimensionless grain diameter
    // d_star = d × ((ρ_s - ρ_f) × g / (ν²))^(1/3)

    if d_star < 4.0 {
        0.24 / d_star  // Fine sediment
    } else if d_star < 10.0 {
        0.14 / d_star.powf(0.64)  // Medium sediment
    } else {
        0.04  // Coarse sediment
    }
}
```

#### Entrainment Condition
```rust
fn is_particle_entrained(bed_shear_stress: f32, particle_diameter: f32,
                          rho_sediment: f32, rho_fluid: f32) -> bool {
    let shields = bed_shear_stress /
                  ((rho_sediment - rho_fluid) * GRAVITY * particle_diameter);
    let shields_critical = critical_shields_stress(particle_diameter);
    shields > shields_critical
}
```

### Practical Implementation Notes

1. **Bed Shear Stress Calculation:**
   ```
   τ = ρ_f × u*²
   ```
   where u* = shear velocity = √(τ/ρ_f)

2. **For Slope Effects:**
   - Critical Shields stress [depends on channel-bed slope](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007JF000831)
   - Steeper slopes → lower threshold (easier entrainment)

3. **Validation:**
   - Shields diagram remains valid primarily for **uniform flow**
   - [Deviations occur](https://www.intechopen.com/chapters/75763) in accelerating/decelerating flows due to vertical particle motion

### Integration with APIC/FLIP

```rust
// During particle update step
for particle in particles.iter_mut() {
    let local_velocity = grid.interpolate_velocity(particle.position);
    let shear_stress = calculate_bed_shear_stress(local_velocity, particle.height_above_bed);

    if is_particle_entrained(shear_stress, particle.diameter,
                             SEDIMENT_DENSITY, FLUID_DENSITY) {
        particle.state = ParticleState::Suspended;
        particle.apply_drag_force(local_velocity);
    } else {
        particle.state = ParticleState::Bedload;
        particle.apply_friction_force();
    }
}
```

**Sources:**
- [Critical Thresholds for Transport and Erosion (HEC-RAS)](https://www.hec.usace.army.mil/confluence/rasdocs/d2sd/ras2dsedtr/6.6/model-description/critical-thresholds-for-transport-and-erosion)
- [Shields formula - Wikipedia](https://en.wikipedia.org/wiki/Shields_formula)
- [Is the critical Shields stress dependent on channel-bed slope?](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007JF000831)
- [Formulae of Sediment Transport in Steady Flows](https://www.intechopen.com/chapters/75763)

---

## 3. Stokes Settling Velocity

### Overview

[Stokes' Law](https://en.wikipedia.org/wiki/Stokes'_law) describes the settling velocity of spherical particles in a viscous fluid under **creeping flow** conditions (low Reynolds number).

### Mathematical Formula

```
v_s = g × (ρ_p - ρ_f) × d² / (18 × μ)
```

**Note:** This formula uses particle **diameter** (d). The equivalent formula using particle **radius** (r, where d = 2r) is:
```
v_s = (2/9) × g × (ρ_p - ρ_f) × r² / μ
```

where:
- v_s = settling velocity (terminal velocity)
- g = gravitational acceleration
- ρ_p = particle density
- ρ_f = fluid density
- d = particle diameter (or r = particle radius)
- μ = dynamic viscosity of fluid

### Validity Conditions

**Critical limitation:** [Particle Reynolds number Re_p < 0.1](https://www.geological-digressions.com/fluid-flow-stokes-law-and-particle-settling/)

```
Re_p = (ρ_f × v_s × d) / μ
```

**Applicable particle sizes:**
- Very fine sand and finer
- Typically d < 100 μm for water at 20°C

### Terminal Velocity Concept

[Balance of forces](https://aerosol.ucsd.edu/SIO217dDepn140114.pdf):
1. **Gravitational force** (downward): F_g = (4/3)πr³ρ_p × g
2. **Buoyant force** (upward): F_b = (4/3)πr³ρ_f × g
3. **Drag force** (upward): F_d = 6πμrv (Stokes drag)

At terminal velocity: F_g = F_b + F_d

### Hindered Settling

For **concentrated suspensions** (high particle volume fraction), particles interact and settle more slowly than isolated particles.

**[Richardson-Zaki equation](https://www.sciencedirect.com/topics/engineering/settling-velocity):**
```
v_h = v_s × (1 - φ)^n
```
where:
- v_h = hindered settling velocity
- v_s = Stokes settling velocity (isolated particle)
- φ = particle volume fraction
- n = empirical exponent (typically 4.65 for Re_p < 0.2)

### Implementation for Non-Spherical Particles

[Correction factors required](https://geoweb.uwyo.edu/geol5330/Dietrich_SettlingVelocity_WRR82.pdf) for non-spherical particles:

**Shape factor (ψ):**
```
ψ = surface_area_sphere / surface_area_particle
```
(where both have same volume)

**Modified Stokes:**
```
v_s = g × (ρ_p - ρ_f) × d² / (18 × μ) × ψ^k
```
where k is empirically determined (typically k ≈ 1)

### Practical Implementation

```rust
fn stokes_settling_velocity(
    particle_diameter: f32,
    particle_density: f32,
    fluid_density: f32,
    dynamic_viscosity: f32,
    shape_factor: f32,  // 1.0 for perfect sphere
) -> f32 {
    let g = 9.81;  // m/s²
    let density_diff = particle_density - fluid_density;

    // Basic Stokes velocity (using diameter)
    let v_stokes = g * density_diff * particle_diameter.powi(2) / (18.0 * dynamic_viscosity);

    // Apply shape correction
    let v_corrected = v_stokes * shape_factor;

    // Verify Reynolds number is in Stokes regime
    let re_p = fluid_density * v_corrected * particle_diameter / dynamic_viscosity;
    if re_p > 0.1 {
        // Use transition regime formula (e.g., Oseen correction)
        return v_stokes * (1.0 + 3.0/16.0 * re_p);
    }

    v_corrected
}

fn hindered_settling_velocity(
    stokes_velocity: f32,
    particle_volume_fraction: f32,
) -> f32 {
    let n = 4.65;  // Richardson-Zaki exponent for Re_p < 0.2
    stokes_velocity * (1.0 - particle_volume_fraction).powf(n)
}
```

### Integration with APIC/FLIP

```rust
// During particle update
for particle in particles.iter_mut() {
    let v_settle = stokes_settling_velocity(
        particle.diameter,
        SEDIMENT_DENSITY,
        WATER_DENSITY,
        WATER_VISCOSITY,
        particle.shape_factor,
    );

    // Apply hindered settling if in concentrated region
    let local_concentration = calculate_local_particle_fraction(particle.position);
    let v_hindered = hindered_settling_velocity(v_settle, local_concentration);

    // Add settling velocity to particle velocity
    particle.velocity.y -= v_hindered * dt;
}
```

### Extended Regimes

For higher Reynolds numbers (Re_p > 0.1), use empirical correlations:

**Intermediate regime (0.1 < Re_p < 1000):**
- Use drag coefficient approach
- C_d varies with Re_p
- Iterative solution required

**Sources:**
- [Stokes Law and particle settling - Geological Digressions](https://www.geological-digressions.com/fluid-flow-stokes-law-and-particle-settling/)
- [Particle Settling Velocity (UCSD)](https://aerosol.ucsd.edu/SIO217dDepn140114.pdf)
- [Settling Velocity of Natural Particles (Dietrich 1982)](https://geoweb.uwyo.edu/geol5330/Dietrich_SettlingVelocity_WRR82.pdf)
- [Stokes' law - Wikipedia](https://en.wikipedia.org/wiki/Stokes'_law)

---

## 4. APIC Boundary Friction and Particle-Grid Transfer

### APIC Overview

[Affine Particle-In-Cell (APIC)](https://dl.acm.org/doi/10.1145/2766996) represents particle velocities as **locally affine** rather than locally constant (as in PIC), enabling:
- Exact conservation of linear and angular momentum
- Dramatically reduced dissipation compared to PIC
- Stable alternative to FLIP without noise

### Particle-Grid Transfer Mechanics

#### Standard PIC Transfer (highly dissipative):
```
v_grid = Σ w_p × v_p  // Grid velocity from particles
v_p_new = Σ w_g × v_grid  // Particle velocity from grid
```

#### APIC Transfer (momentum + affine matrix):
```rust
// Particle to grid
struct Particle {
    velocity: Vec3,
    C: Mat3,  // Affine velocity matrix (stores velocity gradient)
}

fn particle_to_grid_apic(particles: &[Particle], grid: &mut Grid) {
    for cell in grid.cells.iter_mut() {
        cell.velocity = Vec3::ZERO;
        cell.mass = 0.0;
    }

    for particle in particles {
        for (cell_idx, weight) in particle.affected_cells() {
            let x_diff = grid.cell_position(cell_idx) - particle.position;

            // APIC transfer: includes affine correction
            let v_pic = particle.velocity + particle.C * x_diff;

            grid.cells[cell_idx].velocity += weight * particle.mass * v_pic;
            grid.cells[cell_idx].mass += weight * particle.mass;
        }
    }

    // Normalize
    for cell in grid.cells.iter_mut() {
        if cell.mass > 0.0 {
            cell.velocity /= cell.mass;
        }
    }
}

// Grid to particle
fn grid_to_particle_apic(grid: &Grid, particles: &mut [Particle]) {
    for particle in particles.iter_mut() {
        let mut velocity = Vec3::ZERO;
        let mut C = Mat3::ZERO;

        for (cell_idx, weight, gradient) in particle.affected_cells_with_gradients() {
            velocity += weight * grid.cells[cell_idx].velocity;

            // Reconstruct affine matrix
            C += weight * grid.cells[cell_idx].velocity * gradient.transpose();
        }

        particle.velocity = velocity;
        particle.C = C;
    }
}
```

### Boundary Conditions in APIC

[Boundary handling](https://www.cs.ucr.edu/~craigs/papers/2019-mac-apic/paper.pdf) requires special care:

1. **Ghost Cell Extrapolation:**
   - Extrapolate into ghost region BEFORE grid-to-particle transfer
   - Extrapolate AFTER particle-to-grid transfer

2. **Solid Boundary Friction:**

```rust
fn apply_boundary_friction(
    grid: &mut Grid,
    boundary_velocity: Vec3,
    friction_coefficient: f32,
) {
    for cell in grid.boundary_cells.iter_mut() {
        let relative_velocity = cell.velocity - boundary_velocity;
        let normal_velocity = relative_velocity.dot(cell.normal) * cell.normal;
        let tangent_velocity = relative_velocity - normal_velocity;

        // Coulomb friction in tangential direction
        let friction_force = -friction_coefficient *
                            cell.normal_stress *
                            tangent_velocity.normalize();

        // Apply friction impulse
        cell.velocity += friction_force * dt / cell.mass;

        // Enforce no-penetration
        if normal_velocity.dot(cell.normal) < 0.0 {
            cell.velocity -= normal_velocity;
        }
    }
}
```

### Granular Material Contact in APIC

[Novel approaches](https://www.researchgate.net/publication/339087035_Affine_particle_in_cell_method_for_MAC_grids_and_fluid_simulation) address two key issues:

**1. Gap Problem:**
- Material bodies tend to stick at gap ≈ grid spacing
- **Solution:** Combined APIC + specialized grid transfer enforces contact at essentially zero gap

**2. Permanent Sticking:**
- Particles in same grid cell stick permanently
- **Solution:** Iterative momentum update scheme allows natural separation while conserving momentum

### Friction Implementation Strategies

#### Approach 1: Grid-Level Friction (Recommended for APIC)
```rust
fn apply_grid_friction(grid: &mut Grid, dt: f32) {
    for cell in grid.cells.iter_mut() {
        if cell.is_near_boundary() {
            let boundary_normal = cell.get_boundary_normal();
            let v_normal = cell.velocity.dot(boundary_normal) * boundary_normal;
            let v_tangent = cell.velocity - v_normal;

            // Static friction threshold
            let friction_impulse = FRICTION_COEFF * v_normal.length();

            if v_tangent.length() * cell.mass / dt < friction_impulse {
                // Static friction: stop tangential motion
                cell.velocity = v_normal;
            } else {
                // Dynamic friction: reduce tangential velocity
                let friction = -DYNAMIC_FRICTION * friction_impulse *
                               v_tangent.normalize();
                cell.velocity += friction * dt / cell.mass;
            }
        }
    }
}
```

#### Approach 2: Particle-Level Friction (For granular focus)
```rust
fn apply_particle_friction(particles: &mut [Particle], grid: &Grid, dt: f32) {
    for particle in particles.iter_mut() {
        if particle.is_in_contact() {
            let contact_normal = particle.get_contact_normal();
            let relative_velocity = particle.velocity - particle.contact_velocity;

            let v_n = relative_velocity.dot(contact_normal);
            let v_t = relative_velocity - v_n * contact_normal;

            // Coulomb friction
            if v_n < 0.0 {  // In contact
                let normal_force = -v_n * particle.mass / dt;
                let max_friction = FRICTION_COEFF * normal_force;

                let tangent_impulse = v_t * particle.mass / dt;
                if tangent_impulse.length() < max_friction {
                    // Static friction
                    particle.velocity -= v_t;
                } else {
                    // Dynamic friction
                    let friction = -max_friction * v_t.normalize() * dt / particle.mass;
                    particle.velocity += friction;
                }
            }
        }
    }
}
```

### Material Point Method (MPM) Extension

APIC forms the basis of MPM, which [excels at granular materials](https://github.com/larsblatny/matter):

**Matter (open-source MPM)** provides:
- No-slip and frictional boundary conditions
- Material-induced boundary friction (MIBF): uses material's internal friction parameter for terrain interaction
- Drucker-Prager and μ(I)-rheology for granular flow

**Sources:**
- [The affine particle-in-cell method](https://dl.acm.org/doi/10.1145/2766996)
- [Affine particle in cell method for MAC grids](https://www.cs.ucr.edu/~craigs/papers/2019-mac-apic/paper.pdf)
- [Matter: Open-source MPM for granular matter](https://github.com/larsblatny/matter/)
- [Matter v1 paper (GMD 2025)](https://gmd.copernicus.org/articles/10/4367/2017/)

---

## 5. Granular Material Physics: Friction Angles and Repose

### Angle of Repose

The [angle of repose](https://en.wikipedia.org/wiki/Angle_of_repose) is the maximum angle at which granular material remains stable on an inclined surface without sliding.

**Typical values:** 25° to 40° for most granular materials

### Relationship to Friction

[For an inclined plane](https://www.doitpoms.ac.uk/tlplib/granular_materials/repose_angle.php):
```
tan(θ_repose) ≈ μ_s  (coefficient of static friction)
```

However: **θ_repose ≠ angle of internal friction** in general

### Angle of Internal Friction (φ)

**Typical values (from geotechnical data):**
- **Sand:** 30-40° (loose: 30-35°, dense: 35-45°)
- **Gravel:** 35-48°
- **Silt:** 26-35°
- **Clay:** 20°
- **Rock:** ~30°

**Coefficient of internal friction:**
```
μ_internal = tan(φ)
```
For φ = 25-35°: **μ_internal ≈ 0.5-0.7**

### Factors Affecting Angle of Repose

1. **Particle size:** [Smaller particles → higher angle](https://www.pnas.org/doi/10.1073/pnas.2107965118) (more surface area, more friction)
2. **Particle shape:** Angular particles → higher angle (interlocking)
3. **Moisture:** Liquid bridges increase cohesion → higher angle
4. **Surface roughness:** Rougher surfaces → higher angle

### Implementation in DEM/Particle Simulations

#### 1. Measuring Angle of Repose (Funnel Method)
```rust
fn calculate_angle_of_repose(pile_height: f32, pile_base_radius: f32) -> f32 {
    (pile_height / pile_base_radius).atan().to_degrees()
}
```

#### 2. Validating Friction Parameters
```rust
// Calibrate friction coefficient to match target angle of repose
fn calibrate_friction_for_repose_angle(
    target_angle_degrees: f32,
) -> f32 {
    let target_angle_rad = target_angle_degrees.to_radians();
    target_angle_rad.tan()  // Returns coefficient of friction
}

// For sand (typical repose angle 30-35°):
let sand_friction = calibrate_friction_for_repose_angle(32.5);
// Returns: μ ≈ 0.64
```

#### 3. Dynamic Angle of Repose (Rotating Drum)

[Mathematical model](https://www.sciencedirect.com/science/article/pii/S0032591024008209) accounts for:
- Sliding friction coefficient (μ_slide)
- Rolling friction coefficient (μ_roll)
- Froude number (rotational speed effect)

```rust
fn dynamic_repose_angle(
    mu_slide: f32,
    mu_roll: f32,
    froude_number: f32,
) -> f32 {
    // Simplified model
    let base_angle = mu_slide.atan();
    let rolling_correction = mu_roll * froude_number.sqrt();
    (base_angle.tan() + rolling_correction).atan()
}
```

### Granular Flow Rheology: μ(I) Model

For dense granular flows, friction becomes rate-dependent:

```
μ = μ_s + (μ_2 - μ_s) / (I_0 / I + 1)
```

where:
- μ_s = static friction coefficient
- μ_2 = dynamic friction coefficient at high inertial number
- I = inertial number = (γ̇ × d) / √(P/ρ)
- γ̇ = shear rate
- d = particle diameter
- P = pressure
- ρ = density

### Practical Implementation for Sluice Simulation

```rust
struct SedimentParticle {
    friction_static: f32,
    friction_dynamic: f32,
    angle_of_repose: f32,
}

impl SedimentParticle {
    fn new_sand() -> Self {
        Self {
            friction_static: 0.64,   // tan(32.5°)
            friction_dynamic: 0.58,  // ~90% of static
            angle_of_repose: 32.5,
        }
    }

    fn new_silt() -> Self {
        Self {
            friction_static: 0.62,   // tan(31°)
            friction_dynamic: 0.56,
            angle_of_repose: 31.0,
        }
    }

    fn effective_friction(&self, velocity: f32) -> f32 {
        let v_threshold = 0.01;  // m/s
        if velocity < v_threshold {
            self.friction_static
        } else {
            // Smooth transition
            let t = (velocity / v_threshold).min(1.0);
            self.friction_static * (1.0 - t) + self.friction_dynamic * t
        }
    }
}
```

### Packing Fraction and Jamming

**Random close packing:** φ ≈ 0.64 (spheres)
**Critical state:** φ_critical ≈ 0.60 (below which material flows freely)

```rust
fn calculate_local_packing_fraction(
    particle_positions: &[Vec3],
    query_position: Vec3,
    search_radius: f32,
) -> f32 {
    let volume_search = (4.0/3.0) * PI * search_radius.powi(3);
    let mut volume_particles = 0.0;

    for pos in particle_positions {
        if (pos - query_position).length() < search_radius {
            volume_particles += (4.0/3.0) * PI * PARTICLE_RADIUS.powi(3);
        }
    }

    volume_particles / volume_search
}

fn is_jammed(packing_fraction: f32) -> bool {
    packing_fraction > 0.60  // Critical state threshold
}
```

**Sources:**
- [Angle of repose - Wikipedia](https://en.wikipedia.org/wiki/Angle_of_repose)
- [DoITPoMS: Granular Materials](https://www.doitpoms.ac.uk/tlplib/granular_materials/repose_angle.php)
- [An expression for the angle of repose (PNAS)](https://www.pnas.org/doi/10.1073/pnas.2107965118)
- [Dynamic angle of repose in rotating drum](https://www.sciencedirect.com/science/article/pii/S0032591024008209)
- [Geotechnical friction angle data](https://geotechdata.info/parameter/angle-of-friction)

---

## 6. Sluice/Flume Sediment Dynamics

### Overview

Sluice and flume flows exhibit complex sediment behavior involving **entrainment**, **transport**, and **deposition** in a continuous cycle.

### Key Physical Processes

#### 1. Entrainment
- Turbulent eddies remove grains from packed bed
- Governed by Shields criterion (see Section 2)
- Transition from bedload to suspended load

#### 2. Transport
- **Bedload:** Rolling, sliding, saltation near bed
- **Suspended load:** Particles carried by turbulence
- **Washload:** Very fine particles (silt/clay) in permanent suspension

#### 3. Deposition/Packing
- Grains settle from suspension
- Deposit onto packed bed
- Rate controlled by settling velocity and near-bed turbulence

### FLOW-3D Sediment Transport Model

[Commercial implementation](https://www.flow3d.com/modeling-capabilities/sediment-transport-model/) provides practical approach:

**Features:**
- Fully coupled hydrodynamic + sediment transport
- Up to 10 different sediment species with properties:
  - Grain size
  - Mass density
  - Critical shear stress
- Simulates bedload and suspended transport
- Entrainment and erosion for non-cohesive soils

**Key equations:**
```
∂(packed_bed_mass)/∂t = packing_rate - entrainment_rate
∂(suspended_mass)/∂t = entrainment_rate - packing_rate + ∇·(flux)
```

### Particle-Resolved Simulation Approaches

[DNS/grain-resolved simulations](https://escholarship.org/content/qt0wq5b1kk/qt0wq5b1kk_noSplash_887a277d377e5c1dd49fa4bf922a1ec5.pdf) capture full physics:

**Immersed Boundary Method (IBM):**
- Resolves flow around individual particles
- Captures cohesive forces (for fine sediments)
- Accounts for aggregation and break-up
- Computationally expensive but high-fidelity

**CFD-DEM Coupling:**
- Fluid: Volume-averaged Navier-Stokes
- Particles: Discrete Element Method
- Momentum exchange through drag closures

### Rouse Profile for Suspended Sediment

[Vertical concentration distribution](https://en.wikipedia.org/wiki/Rouse_number):

```
C(z) / C_a = ((h - z) / z × (z_a / (h - z_a)))^P
```

where:
- C(z) = concentration at height z
- C_a = reference concentration at height z_a
- h = flow depth
- P = Rouse number = w_s / (κ × u*)
- w_s = settling velocity
- κ = von Kármán constant (≈ 0.4)
- u* = shear velocity

**Rouse Number Interpretation:**
- P < 0.8: Suspended load (uniform distribution)
- 0.8 < P < 2.5: Mixed suspended/bedload
- P > 2.5: Bedload (concentrated near bed)

### Implementation Example

```rust
fn rouse_concentration_profile(
    z: f32,              // Height above bed
    z_ref: f32,          // Reference height
    c_ref: f32,          // Reference concentration
    flow_depth: f32,
    settling_velocity: f32,
    shear_velocity: f32,
) -> f32 {
    let kappa = 0.4;  // von Kármán constant
    let rouse_number = settling_velocity / (kappa * shear_velocity);

    let ratio1 = (flow_depth - z) / z;
    let ratio2 = z_ref / (flow_depth - z_ref);

    c_ref * (ratio1 * ratio2).powf(rouse_number)
}

fn classify_transport_mode(settling_velocity: f32, shear_velocity: f32) -> TransportMode {
    let kappa = 0.4;
    let rouse = settling_velocity / (kappa * shear_velocity);

    if rouse < 0.8 {
        TransportMode::Suspended
    } else if rouse < 2.5 {
        TransportMode::Mixed
    } else {
        TransportMode::Bedload
    }
}

enum TransportMode {
    Suspended,  // Uniform vertical distribution
    Mixed,      // Both suspended and bedload
    Bedload,    // Concentrated near bed
}
```

### Flume Experiment Insights

[Stream flume studies](https://www.science.gov/topicpages/s/settling+velocity+sediment) show:

1. **Velocity dependence:**
   - Lower streamflow → faster deposition
   - Effect is particle-shape dependent

2. **Shape effects:**
   - Non-spherical particles settle differently
   - Accounting for shape reduces settling velocity prediction errors by 3-6%
   - Removes velocity-dependent bias

3. **Concentration effects:**
   - High concentration → hindered settling
   - Muddy slurries develop internal yield stresses
   - Dramatically reduces settling velocity

### Real-Time Implementation Strategy

```rust
struct SedimentLayer {
    bedload_particles: Vec<Particle>,
    suspended_particles: Vec<Particle>,
}

impl SedimentLayer {
    fn update(&mut self, grid: &Grid, dt: f32) {
        // 1. Check entrainment (bed → suspended)
        for particle in self.bedload_particles.iter_mut() {
            let local_shear = grid.get_bed_shear_stress(particle.position);
            if shields_criterion_met(local_shear, particle.diameter) {
                particle.transition_to_suspended();
            }
        }

        // 2. Update suspended particles
        for particle in self.suspended_particles.iter_mut() {
            // Apply drag force
            let local_velocity = grid.interpolate_velocity(particle.position);
            particle.apply_drag(local_velocity);

            // Apply settling
            let settling_vel = stokes_settling_velocity(particle);
            particle.velocity.y -= settling_vel;

            // Check if settling to bed
            if particle.position.y < BED_LEVEL && particle.velocity.y < 0.0 {
                particle.transition_to_bedload();
            }
        }

        // 3. Update bedload (rolling/sliding)
        for particle in self.bedload_particles.iter_mut() {
            let bed_velocity = calculate_bedload_velocity(particle, grid);
            particle.velocity = bed_velocity;
            particle.apply_friction();
        }
    }
}
```

**Sources:**
- [Settling of cohesive sediment: particle-resolved simulations](https://escholarship.org/content/qt0wq5b1kk/qt0wq5b1kk_noSplash_887a277d377e5c1dd49fa4bf922a1ec5.pdf)
- [FLOW-3D Sediment Transport Model](https://www.flow3d.com/modeling-capabilities/sediment-transport-model/)
- [Rouse number - Wikipedia](https://en.wikipedia.org/wiki/Rouse_number)
- [HEC-RAS Rouse-Diffusion Method](https://www.hec.usace.army.mil/confluence/rasdocs/rassed1d/1d-sediment-transport-technical-reference-manual/bed-change/rouse-diffusion-method)

---

## 7. Drag Forces in Two-Phase Flow

### Stokes Drag Regime

For particle Reynolds number Re_p < 0.1:
```
F_drag = 6πμrv
```

### Transition Regime (0.1 < Re_p < 1000)

Use drag coefficient approach:
```
F_drag = 0.5 × C_d × ρ_f × A × v²
```

where C_d varies with Re_p (empirical correlations available)

### CFD-DEM Coupling for Sediment

[Modified Stokes drag models](https://www.sciencedirect.com/science/article/abs/pii/S0032591023003170) for dense flows:
- Account for particle-particle interactions
- Include volume fraction effects
- Blend between dilute (kinetic theory) and dense (μ(I) rheology) regimes

### Practical Implementation

```rust
fn calculate_drag_force(
    particle: &Particle,
    fluid_velocity: Vec3,
    fluid_density: f32,
    dynamic_viscosity: f32,
) -> Vec3 {
    let relative_velocity = fluid_velocity - particle.velocity;
    let speed = relative_velocity.length();

    if speed < 1e-6 {
        return Vec3::ZERO;
    }

    let re_p = fluid_density * speed * particle.diameter / dynamic_viscosity;

    if re_p < 0.1 {
        // Stokes regime
        let drag_coeff = 6.0 * PI * dynamic_viscosity * particle.radius;
        return drag_coeff * relative_velocity;
    } else if re_p < 1000.0 {
        // Transition regime (Schiller-Naumann)
        let cd = 24.0 / re_p * (1.0 + 0.15 * re_p.powf(0.687));
        let area = PI * particle.radius.powi(2);
        let drag_magnitude = 0.5 * cd * fluid_density * area * speed.powi(2);
        return drag_magnitude * relative_velocity.normalize();
    } else {
        // Newton regime
        let cd = 0.44;  // Sphere drag coefficient
        let area = PI * particle.radius.powi(2);
        let drag_magnitude = 0.5 * cd * fluid_density * area * speed.powi(2);
        return drag_magnitude * relative_velocity.normalize();
    }
}
```

**Sources:**
- [Drag force in granular shear flows](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/drag-force-in-granular-shear-flows-regimes-scaling-laws-and-implications-for-segregation/C3FDA5E6EF105A7A74941543C07FD8C7)
- [Modified Stokes law-based drag model](https://www.sciencedirect.com/science/article/abs/pii/S0032591023003170)

---

## 8. Academic References and Advanced Topics

### Key Academic Papers

#### APIC/FLIP for Granular Materials

1. **[Affine Particle-In-Cell Method for MAC Grids](https://www.cs.ucr.edu/~craigs/papers/2019-mac-apic/paper.pdf)**
   - Authors: Xinlei Wang, Yuxing Qiu, Stuart R. Slattery, Yu Fang, Minchen Li, Song-Chun Zhu, Yixin Zhu, Min Tang, Dinesh Manocha, Chenfanfu Jiang
   - Comprehensive APIC analysis for fluid simulation
   - Includes boundary condition handling

2. **[Particle-Based Simulation of Granular Materials (2005)](http://wnbell.com/media/2005-07-SCA-Granular/BeYiMu2005.pdf)**
   - Authors: Nathan Bell, Yizhou Yu, Peter J. Mucha
   - Foundational work on particle-based granular simulation
   - Non-spherical particle representation

3. **[Matter v1: Open-source MPM for Granular Matter (2025)](https://gmd.copernicus.org/articles/10/4367/2017/)**
   - C++ implementation with friction models
   - Drucker-Prager, μ(I)-rheology
   - Material-induced boundary friction

#### Two-Phase Flow Methods

4. **[MultiFLIP for Energetic Two-Phase Fluid Simulation](https://www.cs.ubc.ca/~rbridson/docs/boyd-tog2011-multiflip.pdf)**
   - Authors: Laura Boyd, Robert Bridson
   - FLIP extension for two-phase flows
   - Low dissipation, easy implementation

5. **[SedFoam-2.0: Two-Phase Flow for Sediment Transport](https://gmd.copernicus.org/articles/10/4367/2017/gmd-10-4367-2017.pdf)**
   - OpenFOAM-based solver
   - Implements μ(I) rheology for granular flows
   - Production-ready sediment transport

#### GPU/Real-Time Implementations

6. **[Fast Hydraulic Erosion Simulation on GPU](https://inria.hal.science/inria-00402079/document)**
   - Authors: Xing Mei, Philippe Decaudin, Bao-Gang Hu
   - Real-time erosion using shallow water equations
   - Fully GPU-accelerated

### Industry Implementations

#### Houdini FLIP/APIC
- [FLIP Solver Documentation](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html)
- Supports friction and collision feedback
- Grain solver for granular materials
- Production-proven workflows

#### Commercial Tools
- **FLOW-3D HYDRO:** Industrial sediment transport
- **Insydium NeXus:** GPU-accelerated FLIP/APIC with granular support
- **Unity/Unreal:** Real-time erosion systems for games

---

## 9. Recommended Implementation Roadmap

### Phase 1: Foundation (Current Sprint)
1. Implement Stokes settling velocity for particles
2. Add basic Coulomb friction (static + dynamic)
3. Integrate friction with APIC grid-to-particle transfer

### Phase 2: Sediment Dynamics
1. Implement Shields criterion for entrainment detection
2. Separate particle states: bedload vs suspended
3. Add Rouse profile for vertical concentration distribution
4. Implement hindered settling for concentrated regions

### Phase 3: Advanced Physics
1. μ(I) rheology for dense granular flow
2. Angle of repose validation and calibration
3. Bedload transport (rolling/sliding mechanics)
4. Erosion and deposition mass balance

### Phase 4: Optimization
1. GPU acceleration of particle updates
2. Adaptive particle sampling (more particles where needed)
3. Level-of-detail for distant particles
4. Parallel CFD-DEM coupling if needed

---

## 10. Typical Parameter Values

### Material Properties

#### Sediment Densities
- **Sand:** 2650 kg/m³
- **Silt:** 2700 kg/m³
- **Clay:** 2600 kg/m³
- **Gravel:** 2700 kg/m³

#### Particle Sizes
- **Clay:** < 0.002 mm
- **Silt:** 0.002 - 0.063 mm
- **Fine sand:** 0.063 - 0.2 mm
- **Medium sand:** 0.2 - 0.6 mm
- **Coarse sand:** 0.6 - 2.0 mm
- **Gravel:** > 2.0 mm

#### Friction Coefficients (Dry)
- **Sand:** μ_s = 0.64 (θ = 32.5°), μ_d = 0.58
- **Silt:** μ_s = 0.62 (θ = 31°), μ_d = 0.56
- **Clay:** μ_s = 0.36 (θ = 20°), μ_d = 0.32
- **Gravel:** μ_s = 0.70 (θ = 35°), μ_d = 0.63

#### Critical Shields Stress
- **General:** θ_c ≈ 0.03 - 0.06
- **Sand (d > 1mm):** θ_c ≈ 0.04
- **Fine sediment:** θ_c varies with Re_p

### Fluid Properties (Water at 20°C)

- **Density:** 998 kg/m³
- **Dynamic viscosity:** 0.001 Pa·s (1.0 × 10⁻³)
- **Kinematic viscosity:** 1.0 × 10⁻⁶ m²/s

### Dimensionless Numbers

- **von Kármán constant:** κ = 0.4
- **Richardson-Zaki exponent:** n = 4.65 (Re_p < 0.2)
- **Random close packing:** φ = 0.64
- **Critical packing (jamming):** φ_c = 0.60

---

## 11. Testing and Validation

### Unit Tests

1. **Stokes settling:**
   - Verify terminal velocity matches analytical solution
   - Test Re_p < 0.1 validity check

2. **Shields criterion:**
   - Validate entrainment threshold
   - Test slope correction

3. **Friction:**
   - Verify stick-slip transition
   - Test angle of repose in pile formation

### Integration Tests

1. **Sluice flow:**
   - Particle accumulation in low-velocity zones
   - Erosion in high-velocity zones
   - Mass conservation

2. **Settling column:**
   - Hindered settling for concentrated suspension
   - Rouse profile formation in steady flow

### Benchmark Cases

1. **Dam break with sediment bed**
2. **Rotating drum (repose angle measurement)**
3. **Inclined plane avalanche**
4. **Flume with obstacles (scour and deposition)**

---

## 12. Summary of Key Takeaways

### For Real-Time APIC/FLIP Sediment Simulation:

1. **Use Shields criterion** to determine when particles transition between bedload and suspended states

2. **Implement Stokes settling** for suspended particles, with hindered settling correction for dense regions

3. **Apply Coulomb friction** at boundaries using regularized formulation to avoid discontinuities

4. **Integrate friction with APIC transfers** at grid level for efficiency

5. **Calibrate friction coefficients** to match target angle of repose (sand: ~32°)

6. **Use Rouse number** to classify transport mode and optimize particle updates

7. **Validate against analytical solutions** (Stokes velocity, Shields diagram, angle of repose)

8. **Consider GPU acceleration** for particle updates and drag force calculations

### Critical Equations for Implementation:

```rust
// Settling velocity (using diameter d)
v_s = g × (ρ_p - ρ_f) × d² / (18 × μ)

// Shields parameter
θ = τ / ((ρ_s - ρ_f) × g × d)

// Coulomb friction
F_friction = μ × N

// Hindered settling
v_h = v_s × (1 - φ)^4.65

// Rouse concentration
C(z) ∝ ((h-z)/z)^(w_s/(κ×u*))
```

---

## References

This research synthesis is based on 40+ sources including academic papers, industry documentation, and open-source implementations. All sources are hyperlinked throughout the document.

**Primary academic sources:**
- Matter v1 (GMD 2025) - MPM for granular matter
- APIC papers (Jiang et al., Wang et al.)
- MultiFLIP (Boyd & Bridson 2011)
- SedFoam-2.0 (OpenFOAM)
- Particle-Based Granular Simulation (Bell, Yu, Mucha 2005)

**Industry implementations:**
- Houdini FLIP/Grain solvers (SideFX)
- FLOW-3D sediment transport
- GPU erosion (Mei, Decaudin, Hu)

**Foundational physics:**
- Shields (1936) - Sediment transport criterion
- Stokes (1851) - Settling velocity law
- Coulomb friction models
- Rouse (1937) - Suspended sediment profiles
