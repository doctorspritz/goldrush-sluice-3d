<!-- STATUS: 2D Complete - 3D needs porting (see gap analysis) -->

# Improved Sediment Settling Physics

## Overview

Research and implementation plan for improving non-water particle behavior in the APIC-based sluice simulation to more accurately match real sluice slurry physics. The goal is to achieve realistic density stratification where:
- **Light gangue** (quartz/feldspar sand) stays in suspension and washes out
- **Black sands** (magnetite, ilmenite) drop into riffle pockets
- **Gold** settles fastest into the lowest part of riffle eddies

---

## Design Decision: Clumps vs FLIP Particles (2026-01-09)

> **Decision:** Use `clump.rs` rigid clumps for fine gold in 3D detail zones. Defer flour gold FLIP settling to later.

### Context
The project evolved from 2D APIC (where this plan was written) to 3D FLIP+DEM. We're now at the point of deciding whether to:
- **Option A:** Use FLIP for smaller particles (individual particle settling)
- **Option B:** Scale FLIP coarser so even fine gold is represented as clumps

### Decision Rationale
- `clump.rs` already has shape-based behavior (flat catches in riffles, round rolls)
- This matches real gold behavior in sluices
- Fewer particles = better performance
- Flour gold (< 0.5mm) can be added later if needed

### Gold Size Approach
| Gold Type | Size | Approach | Status |
|-----------|------|----------|--------|
| Picker gold | 2mm+ | Clumps (definite) | clump.rs ready |
| Fine gold | 0.5-2mm | Small clumps | Use clump.rs with smaller templates |
| Flour gold | < 0.5mm | FLIP settling | **Deferred** |

### Current Implementation State
| Component | Purpose | Location |
|-----------|---------|----------|
| `clump.rs` | Rigid gravel clumps (roll/slide/catch) | `crates/sim/src/clump.rs` |
| `dem.rs` | Individual particle DEM (buoyancy, simple drag) | `crates/sim/src/dem.rs` |
| This plan | Ferguson-Church physics-based settling | Implemented in 2D FLIP, needs 3D adaptation |

### TODO: Grid Sizing Analysis
> **Need to plan:** Determine correct FLIP grid coarseness and cell sizing to allow multi-particle clumps in the 3D sluice while maintaining both performance and accuracy.
>
> Key considerations:
> - Real gold flakes: 0.5-2mm typical fine gold
> - Real sluice: ~1.5m long × 0.3m wide
> - Need clumps small enough to show stratification in riffles
> - Need grid coarse enough for real-time performance
> - **Analysis needed:** Can we make it performant AND accurate?

---


## Problem Statement

The current implementation (`flip.rs:450-505`) uses a simplified drag model:
```rust
let beta = BASE_DRAG / density;  // Inverse density scaling
let decay = (-beta * dt).exp(); // Exponential approach to fluid velocity
```

This has several limitations:
1. **No particle size modeling** - settling velocity depends critically on particle diameter
2. **Simplified drag** - doesn't account for Reynolds number regimes (Stokes vs turbulent)
3. **No shape factor** - gold is flaky (slow settling for size), sand is spherical (fast settling)
4. **No hindered settling** - concentrated suspensions slow down ALL particles
5. **Missing slip velocity** - terminal settling velocity not properly computed

## Research Findings

### 1. Ferguson-Church Universal Settling Formula (2004)

The Ferguson-Church equation provides a unified formula across all Reynolds number regimes without iteration:

```
w = (R * g * D²) / (C₁ * ν + √(0.75 * C₂ * R * g * D³))
```

Where:
- `w` = settling velocity (m/s)
- `R` = (ρ_particle - ρ_fluid) / ρ_fluid (relative submerged density)
- `g` = gravitational acceleration
- `D` = particle diameter
- `C₁` = 18 (Stokes constant, increase to 20-24 for angular particles)
- `C₂` = 1.0 for natural sand, 0.4 for spheres, higher for flaky particles
- `ν` = kinematic viscosity

**Key insight**: This formula naturally transitions from Stokes (viscous) to Newton (turbulent) regimes based on particle size.

**Sources:**
- [Ferguson & Church 2004 Paper](https://www.researchgate.net/publication/251601289_A_Simple_Universal_Equation_for_Grain_Settling_Velocity)
- [Python implementation](https://zsylvester.github.io/post/grain_settling/)

### 2. Particle Size Distributions in Real Sluicing

| Material | Typical Size Range | Notes |
|----------|-------------------|-------|
| Coarse gold | 1-5 mm | Easy recovery, settles very fast |
| Fine gold | 0.1-1 mm (100-1000 μm) | Standard sluicing target |
| Flour gold | <0.074 mm (<74 μm) | Stays suspended, difficult recovery |
| Sand/gravel | 0.1-25 mm | Screened to <25mm for sluice |
| Black sand | 0.1-0.5 mm | Indicator mineral, medium settling |

**Key insight**: A 0.1mm gold particle settles at the same rate as a ~1mm sand particle due to density ratio (~7:1).

**Sources:**
- [911 Metallurgist - Flour Gold Recovery](https://www.911metallurgist.com/blog/flour-gold-recovery/)
- [HEC-RAS Particle Settling](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/erosion-and-sediment-transport-under-construction/fall-velocity-and-settling)

### 3. Shape Factor Corrections

Shape affects drag coefficient significantly:

| Material | Corey Shape Factor | C₂ Adjustment | Notes |
|----------|-------------------|---------------|-------|
| Spheres | 1.0 | 0.4 | Theoretical baseline |
| Natural sand | 0.7-0.8 | 1.0 | Slightly angular |
| Magnetite | 0.6-0.7 | 1.1 | Angular crystals |
| Flaky gold | 0.3-0.5 | 1.5-2.0 | Very flat, high drag |

**Key insight**: Gold's flaky shape means it settles slower than a sphere of equal mass - but still MUCH faster than sand due to 7x density difference.

**Sources:**
- [Dietrich 1982 - Shape Effects](https://geoweb.uwyo.edu/geol5330/Dietrich_SettlingVelocity_WRR82.pdf)
- [Shape Factor Research](https://www.researchgate.net/publication/257343308_Dynamic_shape_factor_for_particles_of_various_shapes_in_the_intermediate_settling_regime)

### 4. Hindered Settling (Richardson-Zaki)

When particle concentration is high, ALL particles settle slower:

```
w_hindered = w_clear * (1 - C_v)^n
```

Where:
- `C_v` = volumetric concentration of solids (0 to ~0.6)
- `n` = 4.0 for fine sand, 2.4 for coarse particles

At 10% solids concentration: settling velocity reduced by ~35%
At 30% solids concentration: settling velocity reduced by ~75%

**Key insight**: Heavy slurry (lots of clay) keeps gold in suspension - this is why clear water produces better separation.

**Sources:**
- [Richardson-Zaki Overview](https://www.sciencedirect.com/topics/engineering/hindered-settling)
- [PMC Hindered Settling Model](https://pmc.ncbi.nlm.nih.gov/articles/PMC7514162/)

### 5. APIC-Compatible Two-Phase Coupling

The current architecture is correct for APIC:
1. **Water** → Full APIC (P2G, pressure solve, G2P with C matrix)
2. **Sediment** → Lagrangian with one-way coupling (sample fluid velocity, apply drag)

This matches the [Affine Particle-in-Cell for Two-Phase Flow](https://www.sciencedirect.com/science/article/pii/S2096579621000152) approach where the secondary phase is advected by the primary phase with slip velocity.

**Key insight**: Don't change the APIC framework - improve the Lagrangian sediment forces.

## Proposed Solution

### Phase 1: Add Particle Size and Shape Properties

**File: `particle.rs`**

Add to `ParticleMaterial`:
```rust
/// Particle diameter in simulation units (pixels)
pub fn typical_diameter(&self) -> f32 {
    match self {
        Self::Water => 0.0,        // N/A
        Self::Mud => 0.5,          // Fine clay/silt
        Self::Sand => 2.0,         // Medium sand
        Self::Magnetite => 1.5,    // Black sand
        Self::Gold => 1.0,         // Fine gold (smaller but denser)
    }
}

/// Shape factor C₂ for Ferguson-Church equation
/// Higher = more drag (flaky particles)
pub fn shape_factor(&self) -> f32 {
    match self {
        Self::Water => 1.0,
        Self::Mud => 1.2,          // Irregular clay
        Self::Sand => 1.0,         // Natural sand
        Self::Magnetite => 1.1,    // Angular crystals
        Self::Gold => 1.8,         // Flaky (10:1 aspect ratio)
    }
}
```

### Phase 2: Ferguson-Church Settling Velocity

**File: `flip.rs` - replace `apply_sediment_forces`**

```rust
fn apply_sediment_forces(&mut self, dt: f32) {
    const WATER_DENSITY: f32 = 1.0;
    const GRAVITY: f32 = 150.0;  // pixels/s²
    const KINEMATIC_VISCOSITY: f32 = 1.0;  // Normalized for simulation
    const C1: f32 = 18.0;  // Stokes constant

    self.particles.list.par_iter_mut().for_each(|particle| {
        if !particle.is_sediment() {
            return;
        }

        let density = particle.density();
        let diameter = particle.material.typical_diameter();
        let c2 = particle.material.shape_factor();

        // Ferguson-Church settling velocity
        let r = (density - WATER_DENSITY) / WATER_DENSITY;
        let numerator = r * GRAVITY * diameter * diameter;
        let denominator = C1 * KINEMATIC_VISCOSITY
            + (0.75 * c2 * r * GRAVITY * diameter.powi(3)).sqrt();
        let w_settling = numerator / denominator;

        // Sample fluid velocity
        let v_fluid = particle.old_grid_velocity;

        // Slip velocity: particle sinks relative to fluid
        let slip = Vec2::new(0.0, w_settling);

        // Drag toward (fluid velocity + slip)
        let target = v_fluid + slip;
        let drag_rate = 5.0 / density;  // Inertia scaling
        let blend = (drag_rate * dt).min(1.0);

        particle.velocity = particle.velocity.lerp(target, blend);

        // Safety clamp
        const MAX_VELOCITY: f32 = 100.0;
        let speed = particle.velocity.length();
        if speed > MAX_VELOCITY {
            particle.velocity *= MAX_VELOCITY / speed;
        }
    });
}
```

### Phase 3: Hindered Settling (Optional Enhancement)

When near_density is high, reduce settling velocity:

```rust
// In apply_sediment_forces, after computing w_settling:
let concentration = (particle.near_density / rest_density).min(0.6);
let hindered_factor = (1.0 - concentration).powf(4.0);  // Richardson-Zaki
let w_effective = w_settling * hindered_factor;
```

### Phase 4: Add Size Variation

Add diameter field to Particle struct for natural variation:
```rust
pub struct Particle {
    // ... existing fields ...
    pub diameter: f32,  // Individual particle size
}

// In spawn functions, add random variation:
let base_diameter = material.typical_diameter();
let diameter = base_diameter * (0.7 + rand::random::<f32>() * 0.6);  // ±30%
```

## Acceptance Criteria

### Functional Requirements
- [ ] Gold particles settle faster than sand in still water
- [ ] Flour-sized gold (<0.1 diameter) stays in suspension longer
- [ ] Coarse gold (>2 diameter) drops rapidly into riffle pockets
- [ ] Sand washes downstream while gold accumulates in riffles
- [ ] Black sand (magnetite) settles intermediate - visible in riffle lines
- [ ] High particle concentration slows ALL settling (hindered effect)

### Physics Validation
- [ ] Terminal velocity order: Gold >> Magnetite >> Sand >> Mud
- [ ] Shape factor makes gold settle ~30% slower than a sphere of equal mass
- [ ] Size variation creates natural stratification layers
- [ ] Richardson-Zaki reduction visible at high concentrations

### Performance Requirements
- [ ] No significant FPS drop from new calculations
- [ ] Parallel iteration maintained (rayon)
- [ ] No new memory allocations per frame

## Implementation Phases

### MVP (Phase 1-2) ✅ COMPLETED
1. ✅ Added `typical_diameter()` and `shape_factor()` to ParticleMaterial
2. ✅ Implemented Ferguson-Church `settling_velocity(diameter)` method
3. ✅ Updated `apply_sediment_forces` in flip.rs to use drift-flux model
4. ✅ Added comprehensive tests (14 tests, all passing)
5. ✅ Added `hindered_settling_factor()` function

**Key changes:**
- `particle.rs`: Added ~80 lines (properties, Ferguson-Church formula, tests)
- `flip.rs`: Replaced sediment forces with drift-flux model using Ferguson-Church

**Settling velocities achieved (at typical diameter):**
| Material   | Diameter | Settling | Density | Shape |
|------------|----------|----------|---------|-------|
| Mud        | 0.50     | 2.86 px/s| 2.00    | 1.20  |
| Sand       | 2.00     | 20.83    | 2.65    | 1.00  |
| Magnetite  | 1.50     | 27.86    | 5.20    | 1.10  |
| Gold       | 1.00     | 39.28    | 19.30   | 1.80  |

### Enhanced (Phase 3) ✅ COMPLETED
1. ✅ Added `compute_neighbor_counts()` using spatial hash
2. ✅ Added `neighbor_count_to_concentration()` helper function
3. ✅ Integrated Richardson-Zaki hindered settling into `apply_sediment_forces`
4. ✅ Added 3 new tests for hindered settling

**Hindered settling effect (Gold at 1.0 diameter):**
| Neighbors | Concentration | Factor | Settling (px/s) |
|-----------|---------------|--------|-----------------|
| 4 (dilute)| 10%           | 66%    | 25.93           |
| 8 (normal)| 19%           | 43%    | 16.91           |
| 20 (dense)| 38%           | 15%    | 5.78            |
| 50 (packed)| 49%          | 7%     | 2.59            |

**Key insight**: In concentrated slurry, ALL particles settle ~6-15x slower,
which matches real-world observation that clay-heavy slurry reduces gold recovery.

### Full (Phase 4) ✅ COMPLETED
1. ✅ Added feature flags to `FlipSimulation`:
   - `use_ferguson_church`: Toggle Ferguson-Church vs simple density-based settling
   - `use_hindered_settling`: Toggle Richardson-Zaki correction
   - `use_variable_diameter`: Toggle per-particle vs material-typical diameter
   - `diameter_variation`: Size variation factor (default ±30%)
2. ✅ Added `diameter` field to `Particle` struct with `effective_diameter()` method
3. ✅ Added `with_diameter()` constructor and `spawn_with_variation()` method
4. ✅ Updated spawn methods in `flip.rs` to use diameter variation when enabled
5. ✅ Updated `apply_sediment_forces()` to use feature flags and per-particle diameter

**Feature flags enable runtime comparison of physics modes:**
```rust
sim.use_ferguson_church = true;   // Use Ferguson-Church (default)
sim.use_hindered_settling = true; // Apply Richardson-Zaki (default)
sim.use_variable_diameter = true; // Per-particle diameter (default)
sim.diameter_variation = 0.3;     // ±30% size variation (default)
```

**Key changes:**
- `particle.rs`: Added `diameter` field, `with_diameter()`, `effective_diameter()`, `spawn_with_variation()`
- `flip.rs`: Added 4 feature flags, updated `apply_sediment_forces()` with conditional physics, updated all sediment spawn methods

## References

### Academic Papers
- [Ferguson & Church 2004 - Universal Settling Equation](https://www.researchgate.net/publication/251601289_A_Simple_Universal_Equation_for_Grain_Settling_Velocity)
- [Dietrich 1982 - Natural Particle Settling](https://geoweb.uwyo.edu/geol5330/Dietrich_SettlingVelocity_WRR82.pdf)
- [APIC Two-Phase Flow](https://www.sciencedirect.com/science/article/pii/S2096579621000152)
- [Multi-species MPM Sand/Water](https://math.ucdavis.edu/~jteran/papers/PGKFTJM17.pdf)

### Implementation References
- [Python Grain Settling](https://zsylvester.github.io/post/grain_settling/)
- [HEC-RAS Fall Velocity](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/erosion-and-sediment-transport-under-construction/fall-velocity-and-settling)
- [Engineering LibreTexts - Terminal Velocity](https://eng.libretexts.org/Bookshelves/Civil_Engineering/Slurry_Transport_(Miedema)/04:_The_Terminal_Settling_Velocity_of_Particles/4.04:_Terminal_Settling_Velocity_Equations)

### Gold Recovery Physics
- [911 Metallurgist - Flour Gold](https://www.911metallurgist.com/blog/flour-gold-recovery/)
- [Gravity Concentration in Artisanal Mining](https://mdpi.com/2075-163X/10/11/1026/htm)

## Current Code References

| Component | File | Lines |
|-----------|------|-------|
| Particle struct | `particle.rs` | 165-183 |
| Material properties | `particle.rs` | 96-161 |
| Ferguson-Church settling | `particle.rs` | 125-161 |
| Hindered settling | `particle.rs` | 667-696 |
| Feature flags | `flip.rs` | 56-66 |
| Sediment forces | `flip.rs` | 466-574 |
| Neighbor counts | `flip.rs` | 1230-1272 |
| Spawn with variation | `particle.rs` | 312-335 |
