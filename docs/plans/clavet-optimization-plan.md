# Clavet Near-Pressure Optimization Plan

## Current State

**Bottleneck identified**: `apply_near_pressure` takes 72% of simulation time at 11k particles.

## What Clavet Does in THIS Simulation

Clavet's "double density relaxation" (from the 2005 paper) provides:
1. **Near-pressure forces**: Prevents particles from overlapping at sub-cell scale
2. **Surface tension effects**: Side effect of the pressure model
3. **Smoothing**: Reduces visual noise from particle clumping

Current implementation:
- Runs every 2 frames with 2x dt
- Uses spatial hash for O(n × neighbors) neighbor search
- Two parallel passes: compute densities, then compute forces
- Interaction radius `h = 4.0` cells

## Why This May Be Redundant in FLIP

**Key insight from research**: FLIP already prevents particle clustering through grid-based pressure projection.

From [SideFX Houdini docs](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html):
> "FLIP fluids are useful because particles can be placed on top of each other without destabilizing the system, whereas SPH tends to blow up if you move particles too close."

From [fxguide](https://www.fxguide.com/fxfeatured/the-science-of-fluid-sims/):
> "A great advantage of the FLIP solver over a Particle/SPH fluid sim is how many times you need to evaluate the maths per frame. For a SPH solution to look good the program needs to evaluate it multiple times per frame."

**Clavet was designed for SPH, not FLIP**. In SPH, you NEED near-pressure because there's no grid to enforce incompressibility. In FLIP, the grid projection already handles this.

## The Whole System Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FLIP SIMULATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. classify_cells()     → Mark Solid/Fluid/Air                  │
│ 2. particles_to_grid()  → P2G transfer (velocity to grid)       │
│ 3. store_old_velocities → For FLIP velocity update              │
│ 4. apply_gravity()      → External forces                       │
│ 5. PRESSURE PROJECTION  → Incompressibility (grid-based)        │ ← HANDLES CLUSTERING
│    - enforce_boundary   → Zero velocities at walls              │
│    - compute_divergence → ∇·v                                   │
│    - solve_pressure     → ∇²p = ∇·v (10 Gauss-Seidel iterations)│
│    - apply_gradient     → v -= ∇p                               │
│ 6. grid_to_particles()  → G2P transfer (PIC/FLIP blend)         │
│ 7. apply_sediment_forces→ Drag + buoyancy for sediment          │
│ 7b.NEAR-PRESSURE        → Clavet SPH (72% of time!)             │ ← REDUNDANT?
│ 8. advect_particles()   → Move particles (SDF collision)        │
└─────────────────────────────────────────────────────────────────┘
```

## Options (In Order of Impact)

### Option 1: REMOVE Clavet Entirely (HIGHEST IMPACT)
**Hypothesis**: FLIP's pressure projection already prevents clustering. Clavet is redundant.

**Risk**:
- May see more particle clumping
- May lose surface tension effects
- Need to test visual quality

**How to test**:
1. Comment out `apply_near_pressure` call
2. Run simulation with high particle count
3. Check for: clumping, instability, visual artifacts
4. If needed, add simple weak spring force (O(n) not O(n²))

**Expected result**: 72% speedup if no quality loss.

### Option 2: Replace with Weak Spring Forces
From [SideFX](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html):
> "A weak force between particles that pushes them apart gives better particle distributions."

**Implementation**:
```rust
// Simple O(n) push-apart during advect_particles
if sdf_dist < particle_radius * 2.0 {
    // push particles apart with weak spring
}
```

This is O(n) instead of O(n × neighbors).

### Option 3: Reduce Clavet Frequency
Current: Every 2 frames
Proposed: Every 4-8 frames

**Trade-off**: May see brief clumping between updates.

### Option 4: Reduce Interaction Radius
Current: `h = 4.0` cells (neighbors in 9 cells)
Proposed: `h = 2.0` cells (fewer neighbors)

**Trade-off**: Less smooth pressure distribution.

### Option 5: Particle Reseeding (Bridson)
From [research](https://www.diva-portal.org/smash/get/diva2:441801/FULLTEXT01.pdf):
> "A fluid cell should not have less than three particles and no more than twelve."

Add/remove particles to maintain density instead of computing forces.

### Option 6: Compact Hashing (If Keeping Clavet)
From [University of Freiburg research](https://cg.informatik.uni-freiburg.de/publications/2011_CGF_dataStructuresSPH.pdf):
> "Compact hashing uses a handle array... reducing memory consumption to O(n·k+m)."

Also: Z-curve (Morton code) sorting for cache-friendly iteration.

## Recommendation

**Start with Option 1**: Completely disable Clavet and test.

If visual quality suffers, try Option 2 (weak springs) which is O(n) instead of O(n²).

## Test Plan

1. **Disable Clavet**:
   - Comment out line 126 in flip.rs (`self.apply_near_pressure(dt * 2.0);`)
   - Run with 10k+ particles
   - Check for: particle clumping, explosions, visual artifacts

2. **If issues appear**:
   - Try weak spring force in advect_particles
   - OR reduce Clavet frequency to every 8 frames

3. **Measure**:
   - Sim time at 10k particles
   - Visual quality (record video)
   - Stability over 60+ seconds

## Files to Modify

- `crates/sim/src/flip.rs`:
  - Line 126: Disable/modify `apply_near_pressure` call
  - `advect_particles`: Add weak spring if needed
