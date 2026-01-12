# Plan: Material Hardness & Sediment Transport Fixes

## Current State

### Passing Tests (16/17)
- Collapse/angle of repose physics
- Basic erosion/deposition
- Mass conservation during collapse
- Water physics (Manning friction)

### Failing Tests (2)
1. **`sediment_transport_capacity_velocity_dependent`** - Both 0.8 and 1.6 m/s produce identical suspended sediment (79.53781)
2. **`suspended_sediment_advects_with_flow`** - Advection loses all sediment (418 → 0 cells)

### Ignored Test (1)
- **`material_specific_erosion_rates`** - Currently ignored pending hardness fix

## Root Cause Analysis

### Bug 1: Velocity-Independent Erosion
**Location:** `world.rs:1736`
```rust
let entrain_target_height = entrain_target_height.min(0.5 * dt);
```
The erosion rate is capped at `0.5 * dt` regardless of velocity. Both 0.8 m/s and 1.6 m/s hit this cap, producing identical erosion.

**Fix:** Make the cap proportional to transport capacity:
```rust
let max_erosion = transport_capacity * k_max_erosion * dt;
let entrain_target_height = entrain_target_height.min(max_erosion);
```

### Bug 2: Advection Mass Loss
**Location:** `world.rs:1352-1500` (update_sediment_advection)

The advection function uses complex flux-limiting logic but loses mass. Suspected causes:
- Boundary conditions allowing outflow without accounting
- Concentration update dividing by wrong water depth
- Threshold filtering removing small concentrations

**Fix:** Audit the advection loop for mass conservation:
1. Track total mass before and after each step
2. Ensure boundary fluxes are accounted
3. Remove premature filtering of small concentrations

### Bug 3: Missing Gravel Erosion
**Location:** `world.rs:1738-1776`

Current erosion order: Sediment → Overburden → Paydirt

Gravel is NEVER eroded, even though it sits between overburden and paydirt in the layer stack.

**Fix:** Add gravel erosion between overburden and paydirt:
```rust
// After overburden erosion, before paydirt:
if remaining_demand > 0.0 {
    let available = self.gravel_thickness[idx];
    let resistance = self.params.hardness_gravel; // NEW: add to params
    let take = (remaining_demand / resistance.max(1.0)).min(available);
    self.gravel_thickness[idx] -= take;
    total_eroded_vol += take * cell_area;
    remaining_demand -= take * resistance;
}
```

### Enhancement: Sediment Hardness Parameter
Currently sediment has implicit hardness of 1.0 (no resistance factor).

**Add:** `self.params.hardness_sediment` with default ~0.5 (softest material)

## Implementation Steps

### Step 1: Add Missing Hardness Parameters
File: `world.rs` (WorldParams struct)

```rust
pub struct WorldParams {
    // Existing
    pub hardness_overburden: f32, // default 1.0
    pub hardness_paydirt: f32,    // default 5.0

    // NEW
    pub hardness_sediment: f32,   // default 0.5 (softest)
    pub hardness_gravel: f32,     // default 3.0 (hard)
}
```

### Step 2: Fix Velocity-Dependent Erosion Cap
File: `world.rs:1736`

Replace:
```rust
let entrain_target_height = entrain_target_height.min(0.5 * dt);
```

With:
```rust
// Scale max erosion by transport capacity (proportional to speed²)
let base_max_erosion = 0.1; // m/s max erosion rate at capacity=1
let max_erosion_height = base_max_erosion * transport_capacity.sqrt() * dt;
let entrain_target_height = entrain_target_height.min(max_erosion_height);
```

### Step 3: Add Sediment Hardness to Erosion
File: `world.rs:1741-1748`

Replace:
```rust
// Erode Sediment
if remaining_demand > 0.0 {
    let available = self.terrain_sediment[idx];
    let take = remaining_demand.min(available);
    // ...
}
```

With:
```rust
// Erode Sediment (softest)
if remaining_demand > 0.0 {
    let available = self.terrain_sediment[idx];
    let resistance = self.params.hardness_sediment;
    let take = (remaining_demand / resistance.max(0.1)).min(available);
    self.terrain_sediment[idx] -= take;
    total_eroded_vol += take * cell_area;
    remaining_demand -= take * resistance;
}
```

### Step 4: Add Gravel Erosion
File: `world.rs` (after overburden erosion, ~line 1766)

Add:
```rust
// Erode Gravel (Hard)
if remaining_demand > 0.0 {
    let available = self.gravel_thickness[idx];
    let resistance = self.params.hardness_gravel;
    let take = (remaining_demand / resistance.max(1.0)).min(available);
    self.gravel_thickness[idx] -= take;
    total_eroded_vol += take * cell_area;
    remaining_demand -= take * resistance;
}
```

### Step 5: Fix Advection Mass Conservation
File: `world.rs:1352-1500`

1. Add mass tracking:
```rust
let total_mass_before: f32 = mass_buffer.iter().sum();
// ... advection logic ...
let total_mass_after: f32 = self.suspended_sediment.iter()
    .zip(0..width*depth)
    .map(|(&c, i)| c * self.water_depth(i % width, i / width) * cell_area)
    .sum();
debug_assert!((total_mass_after - total_mass_before).abs() < 0.01 * total_mass_before,
    "Advection lost mass: {} -> {}", total_mass_before, total_mass_after);
```

2. Review boundary handling - ensure outflow at boundaries is tracked
3. Remove the `> 0.0001` threshold that filters small concentrations

### Step 6: Update Tests
File: `terrain_physics.rs`

1. Remove `#[ignore]` from `material_specific_erosion_rates`
2. Add test for gravel erosion rate vs other materials
3. Verify velocity-dependent erosion with wider velocity range

## Hardness Values Reference

| Material | Hardness | Erosion Rate | Physical Basis |
|----------|----------|--------------|----------------|
| Sediment | 0.5 | 2x base | Fine, loose particles |
| Overburden | 1.0 | 1x base | Soil/dirt |
| Gravel | 3.0 | 0.33x base | Coarse, interlocked |
| Paydirt | 5.0 | 0.2x base | Compacted, cohesive |
| Bedrock | ∞ | 0 (immune) | Solid rock |

## Verification

After implementation:
1. `cargo test -p sim3d --test terrain_physics` - all 17 tests pass
2. `cargo run --example river_erosion --release` - visual verification
3. Layer colors change as erosion exposes underlying materials

## Files to Modify

| File | Changes |
|------|---------|
| `crates/sim3d/src/world.rs` | Add hardness params, fix erosion, fix advection |
| `crates/sim3d/tests/terrain_physics.rs` | Remove ignore, add gravel test |
