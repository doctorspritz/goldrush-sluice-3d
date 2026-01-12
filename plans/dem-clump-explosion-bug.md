# Bug: DEM Clumps Explode When Close Together

## Summary

When multiple DEM clumps are spawned close together (e.g., 0.05m apart), some clumps violently explode upward reaching Y positions of 9+ meters when they should settle on a floor at Y=0.075m.

## Reproduction

Run the test:
```bash
cargo test --example washplant_editor test_multigrid_dem_clumps_collide_with_floor -- --nocapture
```

Output shows clumps exploding:
```
Frame 0: Y = [0.485, 0.488, 0.497, 0.506, 0.509]
Frame 30: Y = [0.296, 9.807, 0.431, 0.900, 0.320]  # Clump 1 exploded to Y=9.8!
```

## Root Cause Analysis

### What Works
- Single clump falls and settles correctly on floor (SDF collision works)
- Multiple clumps with wide spacing (0.2m) work correctly
- Gold settles faster than sand test passes (buoyancy/drag physics correct)

### What Fails
- Multiple clumps with close spacing (0.05m) cause explosions
- The explosion happens when clumps land on the floor and their bounding spheres overlap

### The Problem Location

In `crates/sim3d/src/clump.rs`, function `step_dem_internal`, lines 865-873:

```rust
let pre_correction: Vec<Vec3> = self.clumps.iter().map(|clump| clump.position).collect();
self.resolve_dem_penetrations(6);
self.resolve_bounds_positions();
if dt > 0.0 {
    for (clump, prev) in self.clumps.iter_mut().zip(pre_correction) {
        let correction = clump.position - prev;
        clump.velocity += correction / dt;  // <-- PROBLEM: converts position to velocity
    }
}
```

When `resolve_dem_penetrations` pushes overlapping clumps apart, the position delta is converted to velocity by dividing by `dt`. With `dt = 1/60 = 0.0167s`, even a 0.15m position correction becomes a 9 m/s velocity kick.

### Additional Issue

There are multiple places that modify position/velocity without proper limiting:

1. **SDF safety clamp** (lines 829-863): Also pushes clumps and modifies velocity
2. **Inter-clump contact forces** (lines 481-595): Spring-damper forces can be large
3. **The velocity→position→velocity feedback loop**: Position corrections become velocity, which causes more penetration next frame

## Proposed Solution

The DEM solver has a fundamental architecture issue: mixing position-based and impulse-based collision resolution creates instability.

### Option A: Pure Impulse-Based (Recommended)
Remove `resolve_dem_penetrations` position correction and rely purely on spring-damper contact forces. This requires:
1. Proper tuning of `normal_stiffness` and `restitution`
2. Smaller timesteps or substeps for stiff contacts
3. No position-to-velocity conversion

### Option B: Proper Position-Based Dynamics
Use a proper PBD solver that doesn't convert position to velocity:
1. Remove velocity feedback from position corrections
2. Use multiple solver iterations to resolve overlaps
3. Keep positions and velocities decoupled

### Option C: Hybrid with Limiting (Simplest Fix)
Keep current structure but add aggressive limiting:
1. Cap the velocity kick from position corrections (e.g., max 1 m/s per frame)
2. Cap total velocity magnitude more aggressively (e.g., 5 m/s instead of 50 m/s)
3. Add energy tracking to detect and dampen explosions

## Acceptance Criteria

1. **Test Passes**: `test_multigrid_dem_clumps_collide_with_floor` passes with 5 clumps at 0.05m spacing
2. **Clumps Settle**: All clumps must end with Y < 0.3m (started at 0.5m, floor at 0.075m)
3. **No Explosions**: No clump should ever have Y > 1.0m during simulation
4. **Physical Behavior**: Clumps should gently push apart, not violently explode
5. **Existing Tests Pass**: `cargo test -p sim3d` must still pass

## Files to Modify

- `crates/sim3d/src/clump.rs` - Main DEM physics
  - `step_dem_internal()` - The core integration loop
  - `resolve_dem_penetrations()` - Position-based overlap resolution

## Test Commands

```bash
# Run the failing test
cargo test --example washplant_editor test_multigrid_dem_clumps_collide_with_floor -- --nocapture

# Run all DEM tests
cargo test -p sim3d clump

# Run existing physics tests (should still pass)
cargo test -p sim3d
```
