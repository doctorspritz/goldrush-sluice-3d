# Fix Plan: Velocity Being Killed

## Summary of Issues

The fluid simulation has multiple sources of velocity damping that compound to make flow "feel dead":

### Issue 1: Directional Resistance Too Aggressive (my recent addition)
**Location:** `flip.rs:537-564` (grid_to_particles)

The directional resistance applies EVERY frame with:
- `r = 1 + 2 * (1-h_norm)^2` → r=3.0 at bed surface
- `velocity.x /= r^0.35` → dividing by ~1.44 at bed
- `velocity.y /= r` → dividing by 3.0 at bed

**Problem:** This compounds with other damping. At 60fps, after 10 frames at bed:
- Horizontal: velocity reduced to 0.05x original
- Vertical: velocity reduced to 0.000002x original

**Fix:** Either remove entirely OR make it a one-time effect (not per-frame cumulative):
```rust
// Option A: Remove entirely - let the bed SDF collision handle it
// DELETE lines 537-564

// Option B: Apply only when particle is NEWLY near bed
// (would require tracking per-particle state)
```

### Issue 2: Sediment Drags Down Grid Velocity
**Location:** `flip.rs:259` (particles_to_grid)

```rust
let weight_scale = if particle.is_sediment() { 0.5 } else { 1.0 };
```

Sediment particles (especially bedload with velocity=0) contribute weight to grid nodes.
When normalized (`grid.u = sum / weight`), stationary sediment averages velocity toward zero.

**Fix:** Make sediment passive (one-way coupling):
```rust
let weight_scale = if particle.is_sediment() { 0.0 } else { 1.0 };
```

### Issue 3: Near-Pressure Called 3x with Allocations
**Location:** `flip.rs:173-176`

```rust
for _ in 0..3 {
    self.apply_near_pressure(dt / 3.0);
}
```

Inside `apply_near_pressure()`:
- Allocates `Vec<Vec2>` for positions
- Allocates `Vec<(f32,f32)>` for densities
- Allocates `Vec<Vec2>` for forces

**Fix:**
1. Reduce to 1 iteration
2. Pre-allocate buffers in FlipSimulation struct
3. Consider only applying to sediment (water is already incompressible from pressure solve)

```rust
// Change from 3x to 1x
self.apply_near_pressure(dt);
```

### Issue 4: No Vortex Mechanism Active
**Location:** `flip.rs:140-152` (update loop)

The code has `apply_viscosity()` and `apply_vorticity_confinement()` but neither is called.

**Fix:** Add vorticity confinement before pressure solve:
```rust
self.grid.apply_gravity(dt);

// ADD: Vorticity confinement for swirl preservation
if self.use_viscosity {
    self.grid.apply_vorticity_confinement(dt, 40.0); // tunable strength
}

self.grid.enforce_boundary_conditions();
```

### Issue 5: Hindered Settling Broken
**Location:** `flip.rs:apply_sediment_forces()`

- `compute_neighbor_counts()` counts ALL particles in 3x3 cell neighborhood
- At typical density, this returns 36-90 neighbors
- `REST_NEIGHBORS = 8` → concentration saturates → hindered_factor ≈ 0.06
- Settling is crushed to 6% of intended speed

**Fix:** Disable hindered settling by default:
```rust
pub fn new(...) -> Self {
    ...
    use_hindered_settling: false,  // Changed from true
    ...
}
```

### Issue 6: Closed Box - No Outlet
**Location:** `sluice.rs:150-154`

```rust
for j in 0..height {
    sim.grid.set_solid(0, j);
    sim.grid.set_solid(width - 1, j);  // RIGHT WALL BLOCKS FLOW
}
```

Flow has nowhere to go. Pressure equalizes and everything stagnates.

**Fix:** Open the right wall (outlet):
```rust
for j in 0..height {
    sim.grid.set_solid(0, j);
    // DON'T set right wall solid - let water exit
    // sim.grid.set_solid(width - 1, j);
}
```

### Issue 7: SDF Not Computed in All Terrain Builders
**Location:** `sluice.rs`

- `create_sluice()` - calls compute_sdf() ✓
- `create_flat_sluice()` - MISSING
- `create_box()` - MISSING

**Fix:** Add to both functions:
```rust
sim.grid.compute_sdf();
sim.grid.compute_bed_heights();
```

---

## Recommended Fix Order

### Phase 1: Immediate (stop velocity death)
1. **Remove directional resistance** (flip.rs:537-564) - too aggressive
2. **Make sediment passive** (weight_scale = 0.0)
3. **Disable hindered settling** (default false)

### Phase 2: Performance
4. **Reduce near-pressure to 1x** per frame
5. **Pre-allocate near-pressure buffers**

### Phase 3: Enable Vortices
6. **Add vorticity confinement call**
7. **Open right wall** (outlet)

### Phase 4: Cleanup
8. **Fix SDF in all terrain builders**
9. **Re-evaluate directional resistance** with less aggressive parameters IF needed

---

## Expected Behavior After Fixes

1. **Flow speed increases dramatically** - no more averaging with stationary sediment
2. **Vortices form behind riffles** - confinement preserves rotation
3. **Sediment settles at correct rate** - no hindered crushing
4. **Performance improves** - fewer near-pressure passes
5. **Flow looks like a sluice** - water exits right side, recirculates behind riffles
