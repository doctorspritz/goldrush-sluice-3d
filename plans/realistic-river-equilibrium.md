# Plan: Realistic River Erosion with Proper Physics

## Reference Equations (from Fondriest)

### 1. Settling Velocity (Stokes)
```
vs = (g × (ρp - ρf) × Dp²) / 18μ

where:
  vs = settling velocity (m/s)
  g = 9.81 m/s²
  ρp = particle density (~2650 kg/m³ for sand/gravel)
  ρf = fluid density (1000 kg/m³ for water)
  Dp = particle diameter (m)
  μ = dynamic viscosity (0.001 Pa·s for water)
```

**Simplified for our materials:**
| Material | Dp (mm) | vs (m/s) |
|----------|---------|----------|
| Clay | 0.002 | 0.000004 |
| Silt | 0.02 | 0.0004 |
| Fine sand | 0.1 | 0.008 |
| Coarse sand | 0.5 | 0.05 |
| Gravel | 5 | 0.5 (turbulent) |

### 2. Shear Velocity (River)
```
u* = sqrt(g × h × S)

where:
  u* = shear velocity (m/s)
  g = 9.81 m/s²
  h = water depth (m)
  S = bed slope (dimensionless, e.g., 0.01 = 1%)
```

### 3. Bed Shear Stress
```
τ = ρf × u*²

where:
  τ = shear stress (Pa)
  ρf = 1000 kg/m³
  u* = shear velocity
```

### 4. Shields Stress (Dimensionless)
```
τ* = τ / (g × (ρp - ρf) × Dp)

Critical τ* ≈ 0.03-0.06 for most sediments
```

**This is the KEY equation** - transport begins when τ* > τ*_critical

### 5. Bedload Transport (van Rijn, simplified)
```
qb = 0.053 × sqrt((s-1)×g) × d50^1.5 × (T*^2.1 / D*^0.3)

where:
  qb = bedload transport rate (m²/s per unit width)
  s = ρp/ρf ≈ 2.65
  d50 = median particle diameter
  T* = transport stage = (τ* - τ*_crit) / τ*_crit
  D* = dimensionless grain size
```

---

## What Our Current Code Does Wrong

**Current (`world.rs:1693`):**
```rust
let speed = sqrt(vel_x² + vel_z²);  // ❌ Uses flow velocity, not shear velocity
let transport_capacity = speed² × water_depth × 0.5;  // ❌ Made-up formula
if speed > critical_velocity { erode(); }  // ❌ Velocity threshold, not Shields stress
```

**Problems:**
1. Uses surface velocity, not bed shear stress
2. Ignores bed slope entirely
3. No particle size in calculations
4. Transport capacity formula is arbitrary
5. Critical threshold is velocity-based, not stress-based

---

## Implementation Plan

### Step 1: Add Particle Size to Materials

**File:** `crates/sim3d/src/world.rs`

```rust
pub struct WorldParams {
    // ... existing ...

    // Particle sizes (median diameter in meters)
    pub d50_sediment: f32,      // 0.0001 (0.1mm fine silt)
    pub d50_overburden: f32,    // 0.001 (1mm coarse sand)
    pub d50_gravel: f32,        // 0.01 (10mm gravel)
    pub d50_paydirt: f32,       // 0.002 (2mm compacted sand)

    // Physical constants
    pub rho_sediment: f32,      // 2650 kg/m³
    pub rho_water: f32,         // 1000 kg/m³
    pub water_viscosity: f32,   // 0.001 Pa·s
}
```

### Step 2: Calculate Bed Slope

**File:** `crates/sim3d/src/world.rs` in `update_erosion()`

```rust
// Calculate local bed slope from neighboring cells
fn bed_slope(&self, x: usize, z: usize) -> f32 {
    let h_here = self.ground_height(x, z);

    // Get upstream/downstream heights (in flow direction)
    let vel_x = (self.water_flow_x[self.flow_x_idx(x, z)]
               + self.water_flow_x[self.flow_x_idx(x + 1, z)]) * 0.5;
    let vel_z = (self.water_flow_z[self.flow_z_idx(x, z)]
               + self.water_flow_z[self.flow_z_idx(x, z + 1)]) * 0.5;

    // Slope in dominant flow direction
    let slope_x = if vel_x.abs() > 0.01 && x > 0 && x < self.width - 1 {
        (self.ground_height(x - 1, z) - self.ground_height(x + 1, z)) / (2.0 * self.cell_size)
    } else { 0.0 };

    let slope_z = if vel_z.abs() > 0.01 && z > 0 && z < self.depth - 1 {
        (self.ground_height(x, z - 1) - self.ground_height(x, z + 1)) / (2.0 * self.cell_size)
    } else { 0.0 };

    // Return slope magnitude (always positive)
    (slope_x * slope_x + slope_z * slope_z).sqrt().max(0.0001) // Min slope to avoid div by zero
}
```

### Step 3: Calculate Shear Velocity and Stress

**File:** `crates/sim3d/src/world.rs` in `update_erosion()`

```rust
// Replace velocity-based erosion with shear stress
let g = 9.81;
let water_depth = self.water_depth(x, z);
let bed_slope = self.bed_slope(x, z);

// Shear velocity: u* = sqrt(g × h × S)
let shear_velocity = (g * water_depth * bed_slope).sqrt();

// Bed shear stress: τ = ρf × u*²
let shear_stress = self.params.rho_water * shear_velocity * shear_velocity;
```

### Step 4: Calculate Shields Stress for Each Material

```rust
// Get particle size of top material
let (d50, rho_p) = if self.terrain_sediment[idx] > 0.001 {
    (self.params.d50_sediment, self.params.rho_sediment)
} else if self.gravel_thickness[idx] > 0.001 {
    (self.params.d50_gravel, self.params.rho_sediment)
} else if self.overburden_thickness[idx] > 0.001 {
    (self.params.d50_overburden, self.params.rho_sediment)
} else {
    (self.params.d50_paydirt, self.params.rho_sediment)
};

// Shields stress: τ* = τ / (g × (ρp - ρf) × Dp)
let shields_stress = shear_stress / (g * (rho_p - self.params.rho_water) * d50);

// Critical Shields stress (empirical, ~0.03-0.06)
let critical_shields = 0.045;

// Only erode if τ* > τ*_critical
if shields_stress > critical_shields {
    // Calculate transport rate...
}
```

### Step 5: Proper Transport Capacity (van Rijn simplified)

```rust
if shields_stress > critical_shields {
    // Transport stage parameter
    let transport_stage = (shields_stress - critical_shields) / critical_shields;

    // Simplified bedload transport (m²/s per unit width)
    // qb ∝ d50^1.5 × T*^1.5 (simplified from van Rijn)
    let s = rho_p / self.params.rho_water;  // ~2.65
    let qb = 0.05 * ((s - 1.0) * g).sqrt() * d50.powf(1.5) * transport_stage.powf(1.5);

    // Convert to volume per cell per second
    let erosion_rate = qb * self.cell_size;  // m³/s per cell
    let erosion_volume = erosion_rate * dt;
    let erosion_height = erosion_volume / cell_area;

    // Apply erosion (limited by available material)
    // ...
}
```

### Step 6: Proper Settling Velocity

```rust
// Settling velocity: vs = (g × (ρp - ρf) × Dp²) / 18μ
fn settling_velocity(&self, d50: f32) -> f32 {
    let g = 9.81;
    let rho_p = self.params.rho_sediment;
    let rho_f = self.params.rho_water;
    let mu = self.params.water_viscosity;

    // Stokes settling (valid for small particles, Re < 1)
    let vs_stokes = (g * (rho_p - rho_f) * d50 * d50) / (18.0 * mu);

    // For larger particles, use turbulent settling (simplified)
    let vs_turbulent = (4.0 * g * d50 * (rho_p - rho_f) / (3.0 * rho_f * 0.44)).sqrt();

    // Blend based on particle size
    if d50 < 0.0001 { vs_stokes }
    else if d50 > 0.001 { vs_turbulent }
    else { vs_stokes * 0.5 + vs_turbulent * 0.5 }
}

// Apply settling every timestep
let vs = self.settling_velocity(d50_suspended);
let settling_rate = vs / water_depth;  // Fraction that settles per second
let settled_conc = suspended_conc * (1.0 - (-settling_rate * dt).exp());
```

---

## New `update_erosion()` Structure

```rust
pub fn update_erosion(&mut self, dt: f32) {
    let g = 9.81;
    let cell_area = self.cell_size * self.cell_size;

    for z in 1..self.depth-1 {
        for x in 1..self.width-1 {
            let idx = self.idx(x, z);
            let water_depth = self.water_depth(x, z);
            if water_depth < 0.01 { continue; }

            // 1. SETTLING - always happens
            let d50_suspended = self.params.d50_sediment; // Assume fine material in suspension
            let vs = self.settling_velocity(d50_suspended);
            let settling_fraction = (vs * dt / water_depth).min(1.0);
            let settled = self.suspended_sediment[idx] * settling_fraction;
            self.suspended_sediment[idx] -= settled;
            self.terrain_sediment[idx] += settled * water_depth; // Convert conc to height

            // 2. CALCULATE SHEAR STRESS
            let bed_slope = self.bed_slope(x, z);
            let shear_velocity = (g * water_depth * bed_slope).sqrt();
            let shear_stress = self.params.rho_water * shear_velocity * shear_velocity;

            // 3. GET TOP MATERIAL PROPERTIES
            let (d50, available, layer) = self.top_material_properties(idx);

            // 4. SHIELDS STRESS
            let rho_diff = self.params.rho_sediment - self.params.rho_water;
            let shields = shear_stress / (g * rho_diff * d50);
            let critical_shields = 0.045;

            // 5. EROSION (only if τ* > τ*_critical)
            if shields > critical_shields {
                let transport_stage = (shields - critical_shields) / critical_shields;
                let s = self.params.rho_sediment / self.params.rho_water;

                // Transport capacity (van Rijn simplified)
                let qb = 0.05 * ((s - 1.0) * g).sqrt()
                       * d50.powf(1.5)
                       * transport_stage.powf(1.5);

                // Current load vs capacity
                let current_load = self.suspended_sediment[idx] * water_depth;
                let capacity = qb * self.cell_size / water_depth; // As concentration

                if current_load < capacity {
                    // Erosion limited by deficit
                    let deficit = capacity - current_load;
                    let erosion_conc = deficit * 0.1 * dt; // Gradual approach
                    let erosion_height = erosion_conc * water_depth;
                    let actual_erosion = erosion_height.min(available);

                    // Remove from bed, add to suspension
                    self.erode_layer(idx, layer, actual_erosion);
                    self.suspended_sediment[idx] += actual_erosion / water_depth;
                }
            }

            // 6. DEPOSITION (when load > capacity, handled by settling above)
        }
    }
}
```

---

## Parameter Defaults

```rust
impl Default for WorldParams {
    fn default() -> Self {
        Self {
            // ... existing ...

            // Particle sizes (meters)
            d50_sediment: 0.0001,    // 0.1mm fine silt
            d50_overburden: 0.001,   // 1mm coarse sand
            d50_gravel: 0.01,        // 10mm gravel
            d50_paydirt: 0.002,      // 2mm compacted

            // Physical constants
            rho_sediment: 2650.0,    // kg/m³
            rho_water: 1000.0,       // kg/m³
            water_viscosity: 0.001,  // Pa·s
        }
    }
}
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `crates/sim3d/src/world.rs` | New erosion physics, settling, shear stress |
| `crates/game/src/gpu/shaders/heightfield_erosion.wgsl` | GPU version of same physics |
| `crates/sim3d/tests/terrain_physics.rs` | Tests for Shields stress thresholds |

---

## Expected Behavior After Fix

1. **Flat water** (slope ≈ 0) → shear velocity ≈ 0 → no erosion
2. **Fast shallow flow** → high shear → erosion of fine sediment
3. **Gravel exposed** → larger d50 → higher critical shear needed → erosion slows
4. **Channel deepens** → slope decreases → shear decreases → equilibrium
5. **Sediment settles** continuously based on particle size
6. **Load approaches capacity** → erosion rate decreases asymptotically

---

## Key Insight

The **bed slope S** is critical. Our current code ignores it entirely.

- Steep slope (S = 0.05): u* = sqrt(9.81 × 1.0 × 0.05) = 0.7 m/s → strong erosion
- Flat slope (S = 0.001): u* = sqrt(9.81 × 1.0 × 0.001) = 0.1 m/s → weak erosion

As the channel erodes deeper, the slope flattens, automatically reducing erosion power. This is the **natural equilibrium mechanism** we're missing.
