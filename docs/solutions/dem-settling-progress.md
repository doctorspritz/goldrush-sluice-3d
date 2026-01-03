# GPU DEM Settling - Work in Progress

## Problem
Dry particles in the GPU DEM solver exhibit jitter/vibration instead of settling properly. Specific issues:
1. Floor-level particles vibrate instead of coming to rest
2. Mid-air particles have strange falling behavior
3. Gold particles push through sand piles instead of respecting boundaries

## Changes Made

### 1. Jacobi Constraint Averaging (WORKING)
**File:** `crates/game/src/gpu/shaders/dem_forces.wgsl`

Instead of applying collision corrections directly, accumulate them and average:
```wgsl
var accumulated_correction = vec2<f32>(0.0, 0.0);
// ... in collision loop:
accumulated_correction += normal * overlap * my_fraction;
particle_contact_count += 1u;
// ... after loop:
let avg_factor = omega / f32(particle_contact_count);
pos += accumulated_correction * avg_factor;
```
Reference: Macklin et al. "Position Based Simulation Methods" (EG 2015)

### 2. Mass-Weighted Separation for Settled Particles (WORKING)
**File:** `crates/game/src/gpu/shaders/dem_forces.wgsl`

Stationary particles get 50x effective mass, making them resist being pushed:
```wgsl
let j_is_stationary = j_speed_sq < 25.0;  // ~5 px/s
let j_mass_multiplier = select(1.0, 50.0, j_is_stationary);
let effective_mass_j = mass_j * j_mass_multiplier;
```
This prevents gold (density 19.3) from pushing through sand (density 2.65).

### 3. Reduced Collision Radius (WORKING)
**File:** `crates/game/src/gpu/dem.rs` line 869

Changed from 0.5 to 0.35 to pack particles tighter visually:
```rust
radii.push(p.material.typical_diameter() * 0.35 * cell_size);
```

### 4. Sleep Counter System with Support Propagation (PARTIALLY WORKING)
**File:** `crates/game/src/gpu/shaders/dem_forces.wgsl`

Added `sleep_counters` buffer to track consecutive frames of low velocity.
Support propagates from floor upward through the pile:
- Floor contact = immediate support
- Neighbor below with high sleep counter = chain support
- No support = counter decays (can't sleep mid-air)

```wgsl
const SLEEP_THRESHOLD: u32 = 10u;
const JITTER_SPEED_SQ: f32 = 4.0;   // ~2 px/s
const WAKE_SPEED_SQ: f32 = 200.0;   // ~14 px/s

let has_floor_support = floor_contact || near_floor;
let has_chain_support = supported_contacts >= 1u;
let truly_supported = has_floor_support || has_chain_support;

if (!truly_supported) {
    sleep_counter = sleep_counter / 2u;  // Decay mid-air
}
```

### 5. Sand-Then-Gold Test Stage
**File:** `crates/sim/src/stages.rs`

Added `stage_sand_then_gold()` to test layering behavior - pour sand first, then gold on top.

## Remaining Issues

### Floor Vibration
Particles touching the floor still vibrate/jitter instead of settling completely.
- The floor collision pushes particles up (+0.3 margin)
- Gravity pulls them back down
- Creates perpetual micro-bouncing

**Possible fixes to try:**
1. Lock position (not just velocity) for floor-contact particles with high sleep counter
2. Add a "resting contact" state that skips gravity for floor particles
3. Use a softer floor collision with no margin

### Mid-Air Falling Behavior
Particles fall strangely - possibly due to sleep counter logic affecting falling particles.
- The `truly_supported` check might be too aggressive
- Counter decay might be interfering with normal falling

**Possible fixes to try:**
1. Only apply sleep logic after particle has hit something at least once
2. Add explicit "has_landed" flag that persists
3. Separate "falling" state from "settled" state

### Support Propagation
The current implementation checks if neighbor below has high sleep counter, but:
- High sleep counter doesn't necessarily mean "supported"
- Need a cleaner way to propagate support state

**Possible fixes to try:**
1. Use a separate "support level" buffer (0 = unsupported, 1+ = supported)
2. Only floor-contact particles can reach max sleep counter, others capped
3. Two-pass approach: first pass marks floor contacts, second propagates

## Key Files
- `crates/game/src/gpu/dem.rs` - GPU DEM solver setup, buffers, execute()
- `crates/game/src/gpu/shaders/dem_forces.wgsl` - Main DEM compute shader
- `crates/game/examples/sediment_stages_visual.rs` - Visual test (press 9 for sand_then_gold)
- `crates/sim/src/stages.rs` - Stage definitions

## Testing
```bash
cargo run --example sediment_stages_visual -p game --release
# Press 9 for sand_then_gold stage
# Press SPACE to pause, R to reset
```

## References
- Macklin et al. "Position Based Simulation Methods" (Eurographics 2015)
- NVIDIA GPU Gems - Granular simulation
- Müller et al. "Position Based Dynamics" (2007)
