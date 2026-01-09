# 2026-01-09: Sediment Entrainment Drag Model (WIP)

## Session Goal
Fix gravel settling behavior - it was settling too quickly and not entraining in water flow. Needed balance where gravel CAN flow with water but CAN settle in low-flow areas.

## What User Said
- "gravel settling - happens too quickly, doesn't entrain in water flow"
- "need some kind of balance where gravel CAN flow and entrain in water, but CAN become embedded/fill a riffle over time if there's too little water and too much gravel"
- "still not entraining. will continue this later."

## Starting State (from previous session)
- Water heightfield rendering over gravel was just fixed
- FLIP/DEM coupling in place
- Gravel was completely decoupled from vertical FLIP velocity
- CPU-side extra settling acceleration was being applied on top

## What We Did

### 1. Diagnosed the Problem
Read `g2p_3d.wgsl` and found the sediment physics:
- **Horizontal**: Only 40% of FLIP velocity transferred (`fluid_influence = 1/density`)
- **Vertical**: Completely ignored FLIP (`final_y = old_particle_vel.y`)
- Direct gravity applied: `-9.8 * dt`
- CPU also added extra settling in `friction_sluice.rs`

**Root cause**: No drag force from water - gravel just fell with gravity regardless of water velocity.

### 2. Implemented Drag-Based Entrainment Model
New physics in `g2p_3d.wgsl`:

```wgsl
// Drag: blend particle velocity toward water velocity
let drag_rate = sediment_params.drag_coefficient / density;
let drag_blend = min(drag_rate * params.dt, 0.9);
final_velocity = mix(particle_vel, water_vel, drag_blend);

// Buoyancy-reduced gravity
let buoyancy_factor = (density - 1.0) / density;
let effective_gravity = -9.8 * buoyancy_factor;
final_velocity.y += effective_gravity * params.dt;
```

### 3. Added `drag_coefficient` Parameter
- Added to `SedimentParams3D` struct in `g2p_3d.rs`
- Added to `GpuFlip3D` struct in `flip_3d.rs`
- Wired through to shader
- Default: 10.0 (particles approach water velocity at rate of 10/density per second)

### 4. Removed CPU-Side Settling
Removed the extra settling acceleration from `friction_sluice.rs`:
```rust
// REMOVED:
let settling_accel = excess_density * settling_factor * GRAVITY;
clump.velocity.y += settling_accel * dt;
```

### 5. Updated friction_sluice Config
```rust
gpu_flip.sediment_drag_coefficient = 8.0;     // Drag rate (1/s)
gpu_flip.sediment_friction_threshold = 0.05;  // Low - let gravel move
gpu_flip.sediment_friction_strength = 0.1;    // Light friction when slow
```

## Files Changed
- `crates/game/src/gpu/shaders/g2p_3d.wgsl` - New drag model
- `crates/game/src/gpu/g2p_3d.rs` - Added drag_coefficient to SedimentParams3D
- `crates/game/src/gpu/flip_3d.rs` - Added sediment_drag_coefficient field
- `crates/game/examples/friction_sluice.rs` - Removed CPU settling, added drag config
- `crates/game/src/water_heightfield.rs` - (from previous session) sediment bridging

## Dead Ends / Rabbit Holes
None this session - fairly direct path to implementation.

## Regressions
None identified yet.

## Current State: NOT WORKING
Gravel still not entraining properly. Possible issues to investigate:

1. **Drag coefficient too low?**
   - Current: 8.0, try 15-25
   - For density 2.5: effective rate = 8/2.5 = 3.2/s

2. **Gravity still dominates?**
   - Buoyancy factor for 2.5 density = 0.6
   - Effective gravity = 5.88 m/s² (still strong)

3. **FLIP pressure override?**
   - Pressure solver might still push gravel against water flow direction
   - The `new_velocity` sampled from grid might already be wrong

4. **Water velocity at gravel location is low?**
   - Gravel may be in boundary layer where water velocity is near zero
   - Grid velocity interpolation might not capture bulk flow

## Commits
- `09ea043 feat: add drag-based sediment entrainment model (WIP)`

## Next Steps
1. Debug: Print water velocity vs particle velocity to see if drag is even seeing flow
2. Try much higher drag coefficient (20-30)
3. Check if gravel is sampling water velocity correctly (grid cell location)
4. Consider: should gravel use water velocity from slightly above its position?
5. Consider: Shields criterion - only settle when shear stress is low

## Physics Reference
Proper sediment transport involves:
- **Shields criterion**: Entrainment when shear stress > critical threshold
- **Stokes drag**: F = 6πμrv (for spheres in viscous flow)
- **Settling velocity**: Terminal velocity when drag = buoyant weight
- **Bed load vs suspended load**: Different transport regimes

Current model is simplified drag blend - may need more sophisticated approach.
