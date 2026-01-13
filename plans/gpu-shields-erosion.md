# GPU Shields Stress Erosion Implementation Plan

Port Krone-Partheniades erosion physics from CPU (world.rs) to GPU shaders (heightfield_erosion.wgsl).

## Goal

Replace velocity-threshold erosion model in GPU with physically-accurate Shields stress model matching the CPU implementation exactly.

**Current GPU:** Erosion when v² > v_crit²
**New GPU:** Erosion when τ* > τ*_crit using Shields stress (τ* = τ / (g × Δρ × d50))

---

## Implementation Steps

### 1. Extend Params Struct (Rust)

**File:** `crates/game/src/gpu/heightfield.rs`

Expand Params buffer from 12 to 20 u32 slots (80 bytes):

```rust
// Lines 156-163, 1265-1280
let params: [u32; 20] = [
    self.width, self.depth,
    tile_width, tile_depth,
    origin_x, origin_z,
    0, 0,
    bytemuck::cast(self.cell_size),
    bytemuck::cast(dt),
    bytemuck::cast(9.81f32),          // gravity
    bytemuck::cast(0.03f32),          // manning_n
    bytemuck::cast(1000.0f32),        // rho_water
    bytemuck::cast(2650.0f32),        // rho_sediment
    bytemuck::cast(0.001f32),         // water_viscosity
    bytemuck::cast(0.045f32),         // critical_shields
    bytemuck::cast(0.0001f32),        // k_erosion
    bytemuck::cast(0.005f32),         // max_erosion_per_step
    0, 0,                              // padding
];
```

### 2. Update Shader Params (WGSL)

**File:** `crates/game/src/gpu/shaders/heightfield_erosion.wgsl`

**Lines 4-18:** Add new fields to Params struct:

```wgsl
struct Params {
    world_width: u32,
    world_depth: u32,
    tile_width: u32,
    tile_depth: u32,
    origin_x: u32,
    origin_z: u32,
    _pad0: vec2<u32>,
    cell_size: f32,
    dt: f32,
    gravity: f32,
    manning_n: f32,
    rho_water: f32,              // NEW
    rho_sediment: f32,           // NEW
    water_viscosity: f32,        // NEW
    critical_shields: f32,       // NEW
    k_erosion: f32,              // NEW
    max_erosion_per_step: f32,  // NEW
    _pad1: vec2<u32>,
}
```

### 3. Replace Constants (WGSL)

**Lines 22-54:** Remove velocity-based constants, add physics constants:

```wgsl
// Particle sizes (median diameter, meters)
const D50_SEDIMENT: f32 = 0.0001;    // 0.1mm fine silt
const D50_OVERBURDEN: f32 = 0.001;   // 1mm coarse sand
const D50_GRAVEL: f32 = 0.01;        // 10mm gravel
const D50_PAYDIRT: f32 = 0.002;      // 2mm compacted sand

// Hardness multipliers (resistance to erosion)
const HARDNESS_SEDIMENT: f32 = 0.5;
const HARDNESS_OVERBURDEN: f32 = 1.0;
const HARDNESS_GRAVEL: f32 = 2.0;
const HARDNESS_PAYDIRT: f32 = 5.0;

// Turbulent flow friction coefficient
const CF: f32 = 0.003;

// Settling transitions
const D50_STOKES_MAX: f32 = 0.0001;    // < 0.1mm: pure Stokes
const D50_TURBULENT_MIN: f32 = 0.001;  // > 1mm: pure turbulent
const CD_SPHERE: f32 = 0.44;           // Drag coefficient
```

### 4. Add Physics Helper Functions (WGSL)

**After line 82:** Add these functions:

```wgsl
// Settling velocity (Stokes/turbulent blend)
fn settling_velocity(d50: f32, g: f32, rho_p: f32, rho_f: f32, mu: f32) -> f32 {
    let vs_stokes = g * (rho_p - rho_f) * d50 * d50 / (18.0 * mu);
    let vs_turbulent = sqrt(4.0 * g * d50 * (rho_p - rho_f) / (3.0 * rho_f * CD_SPHERE));

    if (d50 < D50_STOKES_MAX) {
        return vs_stokes;
    } else if (d50 > D50_TURBULENT_MIN) {
        return vs_turbulent;
    } else {
        let t = (d50 - D50_STOKES_MAX) / (D50_TURBULENT_MIN - D50_STOKES_MAX);
        return vs_stokes * (1.0 - t) + vs_turbulent * t;
    }
}

// Shear velocity: u* = sqrt(g×h×S + Cf×v²)
fn shear_velocity(depth: f32, slope: f32, vel_x: f32, vel_z: f32, g: f32) -> f32 {
    let grav_term = g * depth * slope;
    let v_sq = vel_x * vel_x + vel_z * vel_z;
    let velocity_term = CF * v_sq;
    return sqrt(grav_term + velocity_term);
}

// Bed shear stress: τ = ρf × u*²
fn shear_stress(u_star: f32, rho_f: f32) -> f32 {
    return rho_f * u_star * u_star;
}

// Shields stress: τ* = τ / (g × (ρp - ρf) × d50)
fn shields_stress(tau: f32, d50: f32, g: f32, rho_p: f32, rho_f: f32) -> f32 {
    let rho_diff = rho_p - rho_f;
    let d50_safe = max(d50, 1e-6);
    return tau / (g * rho_diff * d50_safe);
}

// Get d50 for material type
fn get_d50(mat: u32) -> f32 {
    switch (mat) {
        case 4u: { return D50_SEDIMENT; }
        case 3u: { return D50_OVERBURDEN; }
        case 2u: { return D50_GRAVEL; }
        case 1u: { return D50_PAYDIRT; }
        default: { return 0.0; }
    }
}

// Get hardness for material type
fn get_hardness(mat: u32) -> f32 {
    switch (mat) {
        case 4u: { return HARDNESS_SEDIMENT; }
        case 3u: { return HARDNESS_OVERBURDEN; }
        case 2u: { return HARDNESS_GRAVEL; }
        case 1u: { return HARDNESS_PAYDIRT; }
        default: { return 0.0; }
    }
}
```

### 5. Rewrite update_erosion() Function (WGSL)

**Lines 158-295:** Complete replacement with Shields stress model:

**Key changes:**
- Calculate shear stress from depth, slope, and velocity
- Independent settling (always happens based on vs/h)
- Sequential layer erosion (sediment → gravel → overburden → paydirt)
- Budget tracking with max_erosion_per_step limit
- Each layer uses its own d50 and hardness

**Structure:**
```wgsl
@compute @workgroup_size(16, 16)
fn update_erosion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 1. Check water depth > 0.001m
    // 2. SETTLING (independent)
    //    - Calculate vs = settling_velocity(D50_SEDIMENT, ...)
    //    - Deposit: vs × C × dt / h
    // 3. EROSION (sequential layers)
    //    - Calculate slope, u*, τ, τ*
    //    - For each layer [sediment, gravel, overburden, paydirt]:
    //      * if τ* > τ*_crit:
    //        - excess = (τ* - τ*_crit) / τ*_crit
    //        - rate = k_erosion × excess / hardness
    //        - erode min(rate×dt, available, budget)
    //        - update budget
    // 4. Add eroded material to suspension
}
```

### 6. Remove Obsolete Functions (WGSL)

**Lines 136-156:** Delete:
- `get_critical_velocity()` - replaced by Shields stress
- `get_erosion_rate()` - replaced by physics calculation

**Keep:**
- `compute_surface_material()` - still needed
- `get_idx()`, `get_ground_height()`, `get_terrain_slope()` - utilities

---

## Validation

### Test Framework

**Create:** `crates/game/tests/gpu_erosion_validation.rs`

```rust
#[test]
fn test_gpu_cpu_erosion_match() {
    // Setup identical CPU World and GPU Heightfield
    // Run one erosion step (dt=0.1)
    // Compare results cell-by-cell
    // Assert < 0.1% relative error
}
```

**Test cases:**
1. Low Shields stress (τ* < 0.045) → no erosion
2. High Shields stress → erosion occurs
3. Multi-layer erosion → sequential removal
4. Settling in still water
5. Combined erosion + settling

### Visual Validation

Run `cargo run --example detail_zone --release` and verify:
- Terrain erosion looks realistic
- No artifacts or NaN values
- Channels deepen gradually
- Settling ponds form correctly

---

## Key Physics Formulas

**Shields Stress:**
```
τ* = τ / (g × (ρp - ρf) × d50)
```

**Shear Stress:**
```
τ = ρf × u*²
where u* = √(g × h × S + Cf × v²)
```

**Erosion Rate:**
```
E = k_erosion × (τ* - τ*_crit) / τ*_crit / hardness  (when τ* > τ*_crit)
```

**Settling Velocity:**
```
Stokes:    vs = g × (ρp - ρf) × d² / (18 × μ)     (d < 0.1mm)
Turbulent: vs = √(4 × g × d × (ρp - ρf) / (3 × ρf × Cd))  (d > 1mm)
Blended:   linear interpolation (0.1mm ≤ d ≤ 1mm)
```

---

## Critical Files

1. **crates/game/src/gpu/shaders/heightfield_erosion.wgsl** - Main implementation
2. **crates/game/src/gpu/heightfield.rs** - Params extension
3. **crates/sim3d/src/world.rs** - CPU reference (lines 698-774, 1867-1990)
4. **crates/game/tests/gpu_erosion_validation.rs** - New test file
5. **crates/game/examples/detail_zone.rs** - Visual validation

---

## Expected Effort

- Params extension: 30 min
- Shader struct update: 15 min
- Constants: 15 min
- Helper functions: 1 hour
- update_erosion() rewrite: 1 hour
- Cleanup: 15 min
- Testing: 2 hours
- Validation: 1 hour

**Total: ~6-7 hours**

---

## Success Criteria

✅ Shader compiles without errors
✅ No GPU crashes or NaN values
✅ Erosion occurs when τ* > τ*_crit
✅ No erosion when τ* < τ*_crit
✅ CPU/GPU results match < 0.1% error
✅ Sequential layer erosion works correctly
✅ Visual behavior is realistic
✅ Performance < 1ms for 256x256 grid

---

## Material Properties Reference

| Material   | d50 (m) | Hardness | Description      |
|------------|---------|----------|------------------|
| Sediment   | 0.0001  | 0.5      | Fine silt        |
| Overburden | 0.001   | 1.0      | Coarse sand      |
| Gravel     | 0.01    | 2.0      | Gravel           |
| Paydirt    | 0.002   | 5.0      | Compacted sand   |
| Bedrock    | N/A     | ∞        | Does not erode   |
