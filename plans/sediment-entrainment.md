<!-- STATUS: Work In Progress -->

# Sediment Entrainment Plan

## Goal

Allow deposited sediment to be re-entrained (picked back up) when flow velocity exceeds a critical threshold. This completes the sediment transport cycle: suspension → settling → deposition → entrainment.

## Physics Background

### Shields Parameter

The Shields parameter determines when sediment begins to move:

```
τ* = τ_b / ((ρ_s - ρ_w) * g * d)

Where:
- τ_b = bed shear stress
- ρ_s = sediment density (2650 kg/m³ for sand)
- ρ_w = water density (1000 kg/m³)
- g = gravity
- d = particle diameter
```

Critical Shields number τ*_c ≈ 0.03-0.06 for sand.

### Bed Shear Stress

For channel flow:
```
τ_b = ρ_w * u*²

Where u* = friction velocity ≈ κ * u / ln(z/z0)
```

Simplified for game: use velocity magnitude at cell as proxy for shear.

## Implementation Plan

### Step 1: Track Deposited Cell Ages

Add `deposit_time: Vec<u32>` to grid - frames since deposition. Fresh deposits are more easily entrained.

### Step 2: Sample Flow Velocity at Deposits

For each deposited cell, sample grid velocity just above it:
```rust
let vel_above = grid.sample_velocity(Vec2::new(x + 0.5, y - 0.5));
let speed = vel_above.length();
```

### Step 3: Compute Entrainment Threshold

```rust
const CRITICAL_VELOCITY: f32 = 15.0;  // cells/frame - tune empirically
const ENTRAINMENT_RATE: f32 = 0.1;    // probability per frame when exceeded
```

### Step 4: Spawn Entrained Particles

When velocity exceeds threshold:
1. Probabilistically remove deposited cell
2. Spawn N sand particles at that location
3. Give particles initial velocity matching flow + small random component

### Step 5: Update SDF

After removing deposited cells, recompute SDF in affected region.

## Code Location

New function in `flip.rs`:
```rust
fn entrain_deposited_sediment(&mut self, dt: f32) {
    // For each deposited cell...
    // Check velocity above
    // Probabilistic entrainment
    // Spawn particles, clear cell
}
```

Call after `deposit_settled_sediment()` in update loop.

## Tuning Parameters

| Parameter | Initial Value | Effect |
|-----------|---------------|--------|
| CRITICAL_VELOCITY | 15.0 | Higher = more stable deposits |
| ENTRAINMENT_RATE | 0.1 | Higher = faster erosion |
| PARTICLES_PER_CELL | 4 | Matches deposition count |
| AGE_FACTOR | 0.01 | Older deposits harder to move |

## Testing

1. Build up deposits at low flow
2. Increase inlet velocity (→ key)
3. Observe deposits eroding from upstream
4. Verify particles re-enter suspension

## Edge Cases

- Don't entrain original terrain (only `is_deposited()` cells)
- Rate-limit entrainment to avoid sudden SDF changes
- Consider neighbor stability (isolated cells easier to entrain)
