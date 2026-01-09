# Post-Mortem: Particle Floor Collision Debugging (Failed)

**Date:** 2026-01-08  
**Status:** ❌ Failed  
**Objective:** Fix particles falling through terrain and implement interactive spawning

## What We Tried

### 1. Floor Collision Fix
**Problem:** Particles were falling through the terrain floor.

**Attempted Solution:**
- Removed density check (`if (density > 1.0)`) from `sdf_collision_3d.wgsl` that was restricting floor collision to sediment only
- Updated `bed_heights` buffer to include full terrain surface (bedrock + overburden + paydirt) instead of just bedrock

**Files Modified:**
- `crates/game/src/gpu/shaders/sdf_collision_3d.wgsl` (lines 142-153)
- `crates/game/examples/tailings_pond.rs` (lines 467-471)

### 2. Interactive Mouse-Based Spawning
**Problem:** Needed better control over where particles spawn.

**Attempted Solution:**
- Implemented raycasting from mouse cursor against terrain
- Particles spawn 20m above the raycast hit point
- Added view/projection matrix calculation for proper screen-to-world transformation

**Files Modified:**
- `crates/game/examples/tailings_pond.rs` (lines 385-461)

### 3. Shader Uniform Alignment
**Problem:** Potential rendering issues due to struct padding mismatches.

**Attempted Solution:**
- Updated `particle_3d.wgsl` `Uniforms` struct to exactly match Rust definition
- Added all fields: `cell_size`, `grid_width`, `grid_depth`, `time`, `padding`

**Files Modified:**
- `crates/game/src/gpu/shaders/particle_3d.wgsl` (lines 4-12)

### 4. Stream-Based Spawning
**Problem:** Clumped particles made behavior hard to observe.

**Attempted Solution:**
- Reduced particles per frame from 30 to 5
- Reduced spawn radius from 0.2 to 0.1
- Reduced spread from 0.1 to 0.05

**Files Modified:**
- `crates/game/examples/tailings_pond.rs` (lines 432-461)

## Why It Failed

### Root Causes (Suspected)
1. **FLIP 3D Physics Issues:**
   - The 3D FLIP simulation may have fundamental stability problems
   - Pressure solver might not be converging properly
   - Grid-to-particle (G2P) transfer could be corrupting particle states
   - Particle-to-grid (P2G) scatter/gather might have race conditions

2. **Collision Detection Problems:**
   - The `sdf_collision_3d.wgsl` shader might be executing too late in the pipeline
   - Particles might be getting NaN velocities or positions before collision check
   - The `bed_height` sampling might be incorrect (wrong indexing, wrong coordinate space)

3. **Rendering vs Simulation Mismatch:**
   - Particles might be simulating correctly but rendering incorrectly
   - The billboard shader might be culling particles unintentionally
   - Depth testing might be causing particles to disappear

4. **Architecture Mismatch:**
   - Trying to bridge 3D particles with a 2.5D heightfield is fundamentally problematic
   - The `GpuBridge3D` absorption/emission logic might be interfering with particle lifetimes
   - The heightfield and FLIP simulation might be fighting over particle ownership

## What We Learned

1. **3D FLIP is Complex:** The full 3D FLIP implementation has many moving parts (P2G, pressure solve, G2P, advection, collision) and debugging requires isolating each stage.

2. **Visualization is Critical:** Without proper debug visualization (particle velocities, grid pressures, cell types), it's nearly impossible to diagnose what's going wrong.

3. **Simpler is Better:** The hybrid 3D particles + 2.5D heightfield approach adds significant complexity. A pure 2.5D approach or pure 3D approach would be easier to debug.

4. **Need Unit Tests:** The GPU compute shaders have no unit tests. We're flying blind.

## Recommendations for Future Work

### Short Term (If Continuing This Approach)
1. **Add Debug Visualization:**
   - Render particle velocities as arrows
   - Render grid cell types as colored voxels
   - Render pressure field as heatmap
   - Add particle count display

2. **Isolate Stages:**
   - Test P2G in isolation (spawn particles, check grid values)
   - Test pressure solver in isolation (set divergence, check pressure)
   - Test G2P in isolation (set grid velocities, check particle updates)
   - Test collision in isolation (spawn particles below floor, verify bounce)

3. **Simplify:**
   - Remove sediment physics temporarily
   - Remove vorticity confinement
   - Remove Drucker-Prager model
   - Just get basic water particles falling and bouncing

### Long Term (Alternative Approaches)
1. **Pure 2.5D Shallow Water:**
   - Use heightfield for everything (water + sediment)
   - Much simpler, proven to work
   - Can still look good with proper rendering

2. **Pure 3D Voxel-Based:**
   - Full 3D grid simulation (no particles)
   - Use marching cubes for surface extraction
   - More memory intensive but simpler logic

3. **Hybrid with Clear Boundaries:**
   - 2.5D heightfield for "background" simulation
   - 3D particles ONLY in small "active zones" (e.g., 10x10x10 cells)
   - Clear handoff between systems (no continuous bridge)

## Files to Revert (If Abandoning)
- `crates/game/src/gpu/shaders/sdf_collision_3d.wgsl`
- `crates/game/src/gpu/shaders/particle_3d.wgsl`
- `crates/game/examples/tailings_pond.rs`

## Conclusion

The particle floor collision issue is symptomatic of deeper problems in the 3D FLIP implementation. Without proper debugging tools and a clearer architecture, continuing to patch individual issues is not productive. 

**Recommendation:** Pause 3D particle work and focus on the proven 2.5D heightfield approach, or invest in proper debugging infrastructure before continuing.
