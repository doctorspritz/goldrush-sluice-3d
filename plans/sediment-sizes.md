# Plan: Varying Sediment Sizes

## Current State (Already Implemented!)
- `Particle.diameter` field exists
- `spawn_with_variation()` creates particles with ±30% size variation
- `use_variable_diameter: true` by default
- `settling_velocity(diameter)` uses Ferguson-Church equation
- Settling physics already accounts for particle size!

## What's Missing
1. **Visual rendering** - uses `render_scale()` (returns 1.0) instead of `diameter`
2. **Shape properties** - sphericity/roughness for future extension

## Implementation

### Step 1: Add shape properties to Particle
```rust
// In particle.rs - Particle struct
pub sphericity: f32,  // 1.0 = perfect sphere, 0.3 = flaky/flat
pub roughness: f32,   // 0.0 = smooth, 1.0 = very rough
```

Default values from material:
- Water: N/A
- Sand: sphericity=0.8, roughness=0.3 (rounded grains)
- Magnetite: sphericity=0.6, roughness=0.5 (angular crystals)
- Gold: sphericity=0.3, roughness=0.2 (flat flakes, smooth surface)

### Step 2: Update rendering to use diameter
In `render.rs`, change:
```rust
// FROM:
let size = base_size * particle.material.render_scale();

// TO:
let diameter_scale = particle.effective_diameter() / 2.0; // normalize to typical range
let size = base_size * diameter_scale.clamp(0.5, 2.0);
```

### Step 3: Add keyboard toggle in game
- Key `M` to toggle variable size rendering ON/OFF
- Shows in UI: "SIZE VAR: ON/OFF"

## Files Changed
1. `crates/sim/src/particle.rs` - add sphericity, roughness fields
2. `crates/game/src/render.rs` - use diameter for visual size
3. `crates/game/src/main.rs` - keyboard toggle

## Future Extensions (shape effects on physics)
- Sphericity affects drag coefficient (flatter = more drag)
- Roughness affects friction (rougher = more bed friction)
- Flaky particles orient in flow (gold behavior)
