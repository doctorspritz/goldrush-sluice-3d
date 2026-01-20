# Sediment Physics Model

This document describes the physics model for sediment transport in the FLIP simulation.

---

## Overview

Sediment particles are simulated alongside water using a unified FLIP solver. The key difference is that sediment particles have:
- Higher density (ρ > 1.0)
- Settling velocity (gravity-driven sinking)
- Friction when slow-moving
- Reduced entrainment (harder to pick up)

---

## Particle Properties

```rust
struct Particle3D {
    position: Vec3,
    velocity: Vec3,
    density: f32,    // 1.0 = water, 2.65 = sand, 19.3 = gold
    c_matrix: Mat3,  // APIC affine velocity
}
```

### Material Densities

| Material | Density | Settling Velocity | Notes |
|----------|---------|-------------------|-------|
| Water | 1.0 | 0 | Reference |
| Clay | 1.6 | 0.001 m/s | Cohesive, slow settling |
| Sand | 2.65 | 0.05 m/s | Medium settling |
| Gravel | 2.0 | 0.1 m/s | Fast settling |
| Black Sand | 2.8 | 0.08 m/s | Indicator mineral |
| Gold | 19.3 | 0.15 m/s | Very fast settling |

---

## G2P Sediment Model

The sediment physics are applied in `g2p_3d.wgsl` after the standard FLIP velocity update.

### Parameters (`SedimentParams3D`)

```rust
pub struct SedimentParams3D {
    settling_velocity: f32,     // Base settling speed (m/s)
    friction_threshold: f32,    // Speed below which friction activates (m/s)
    friction_strength: f32,     // Damping factor when slow (0-1 per frame)
    vorticity_lift: f32,        // How much vorticity suspends sediment
    vorticity_threshold: f32,   // Minimum vorticity to lift
    drag_coefficient: f32,      // Rate of velocity alignment with water (1/s)
    // Gold-specific parameters
    gold_density_threshold: f32,
    gold_drag_multiplier: f32,
    gold_settling_velocity: f32,
    gold_flake_lift: f32,
}
```

### Physics Steps (in G2P)

1. **Standard FLIP/PIC Blend**
   ```
   v_new = v_pic + flip_ratio * (v_flip - v_pic)
   ```

2. **Settling Velocity**
   ```
   v.y -= settling_velocity * (density - 1.0) * dt
   ```
   Heavier particles sink faster.

3. **Drag Toward Water Velocity**
   ```
   v += drag_coefficient * (v_water - v) * dt / density
   ```
   Particles are entrained by surrounding water. Heavier particles resist more.

4. **Vorticity Lift**
   ```
   if vorticity_mag > vorticity_threshold:
       v.y += vorticity_lift * vorticity_mag * dt
   ```
   Turbulent regions suspend sediment.

5. **Friction When Slow**
   ```
   if speed < friction_threshold:
       v *= (1.0 - friction_strength)
   ```
   Slow particles decelerate quickly (simulates bed friction).

---

## Sediment Pressure (Drucker-Prager)

For granular materials under compression, we compute overburden pressure.

### Overburden Calculation (`sediment_pressure_3d.wgsl`)

```
For each (x, z) column, scan top-to-bottom:
    accumulated_pressure = 0
    for j from top to bottom:
        cell_mass = sediment_count[i,j,k] * particle_mass
        effective_weight = cell_mass * gravity * buoyancy_factor
        accumulated_pressure += effective_weight / cell_area
        sediment_pressure[i,j,k] = accumulated_pressure
```

### Buoyancy Factor

```
buoyancy_factor = 1 - ρ_water / ρ_sediment
               ≈ 0.62 for sand (ρ = 2.65)
```

Submerged sediment is lighter due to buoyancy.

### Usage (Currently Disabled)

The sediment pressure was intended for Drucker-Prager yield:
- High-pressure cells would resist deformation
- Packed sediment would behave as solid
- This caused jamming issues and is disabled

---

## Sediment Fraction

Each cell tracks its sediment/water ratio:

```wgsl
sediment_fraction = sediment_count / (water_count + sediment_count)
```

### Uses

1. **Porosity Drag**: High-sediment cells slow water flow
2. **Rendering**: Color particles by sediment fraction
3. **Diagnostics**: Track sediment distribution

---

## Porosity Drag

Water velocity is reduced in cells with high sediment concentration.

```wgsl
// porosity_drag_3d.wgsl
porosity = 1.0 - sediment_fraction * max_reduction
grid_velocity *= porosity
```

This models how packed sediment creates flow resistance.

---

## Bed Exchange (Deposition/Entrainment)

### Shields Parameter

The Shields parameter determines whether sediment moves:

```
τ* = τ_bed / ((ρ_s - ρ_w) * g * d)

where:
  τ_bed = bed shear stress (from water velocity)
  ρ_s = sediment density
  ρ_w = water density
  g = gravity
  d = grain diameter
```

### Critical Shields Number

```
τ*_crit ≈ 0.047 (sand)
         ≈ 0.03  (gravel)
```

- If τ* > τ*_crit: erosion (entrainment)
- If τ* < τ*_crit: deposition

### Bed Flux Calculation (`bed_flux_3d.wgsl`)

```
shear_stress = ρ_water * friction_coeff * velocity²
shields = shear_stress / ((ρ_sed - ρ_water) * g * diameter)

if shields > shields_critical:
    flux = entrainment_coeff * (shields - shields_critical)^1.5
else:
    flux = -settling_rate * sediment_near_bed
```

---

## Gold Separation Physics

Gold separates from lighter material due to its extreme density (19.3).

### Key Mechanisms

1. **Differential Settling**
   Gold sinks 3x faster than sand in still water.

2. **Hindered Settling**
   In dense slurry, gold pushes through while lighter particles are blocked.

3. **Riffle Trapping**
   In sluices, riffles create low-velocity zones where gold settles but sand is washed over.

4. **Vorticity Rejection**
   Gold resists suspension by vorticity due to high inertia.

### Gold-Specific Parameters

```rust
gold_density_threshold: 10.0,  // Above this = gold
gold_drag_multiplier: 0.5,     // Gold resists entrainment
gold_settling_velocity: 0.15,  // Fast settling
gold_flake_lift: 0.0,          // Flaky gold can lift (optional)
```

---

## Typical Parameter Values

### Sluice Simulation
```rust
SedimentParams3D {
    settling_velocity: 0.05,      // Sand settling
    friction_threshold: 0.1,      // Start friction below 0.1 m/s
    friction_strength: 0.3,       // 30% damping per frame when slow
    vorticity_lift: 1.5,          // Moderate lift in turbulence
    vorticity_threshold: 2.0,     // Need moderate vorticity to lift
    drag_coefficient: 10.0,       // Moderate entrainment
    gold_density_threshold: 10.0,
    gold_drag_multiplier: 0.3,    // Gold resists entrainment
    gold_settling_velocity: 0.15,
    gold_flake_lift: 0.0,
}
```

### Pure Water (No Sediment Physics)
```rust
SedimentParams3D::default()  // All zeros except drag
```

---

## References

- Shields, A. (1936). "Application of similarity principles and turbulence research to bed-load movement."
- Meyer-Peter, E. & Müller, R. (1948). "Formulas for bed-load transport."
- Drucker, D.C. & Prager, W. (1952). "Soil mechanics and plastic analysis or limit design."
