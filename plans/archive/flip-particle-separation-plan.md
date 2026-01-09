# FLIP-Native Particle Separation Plan

## Problem Statement

Without Clavet near-pressure, particles compress/flatten to floor. FLIP's grid-based pressure projection prevents velocity divergence but doesn't prevent particles from physically occupying the same space.

**We need particle-level separation, but Clavet (SPH technique) is wrong tool for FLIP.**

## Research Findings

### What Production FLIP Solvers Use

From [Houdini FLIP Solver docs](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html):

1. **Separation Relaxation** (Push Apart)
   - Simple spring-like force between overlapping particles
   - Uses spatial grid for O(n) neighbor lookup
   - Parameters: iterations, rate, scale
   - Quote: "Despite the velocity projection stage, particles can end up closer together than their pscale"

2. **Reseeding**
   - Maintain target particles per cell (Bridson: 3-12, optimal: 8)
   - Add particles when below threshold
   - Remove particles when above threshold
   - Quote: "Particles Per Voxel sets target particle density per cell"

3. **Particle Radius**
   - Each particle has defined radius (pscale)
   - Separation based on radius overlap, not SPH kernel

### What Clavet Does vs What We Need

| Clavet (SPH) | FLIP-Native |
|--------------|-------------|
| Double density relaxation | Simple push apart |
| Computes pressure from density | Just checks overlap |
| O(n × neighbors) per particle | O(n) with grid |
| Designed for pure SPH | Designed for PIC/FLIP |
| Smooth pressure field | Binary overlap check |

### Key Quote from [Matthias Müller](https://matthias-research.github.io/pages/tenMinutePhysics/18-flip.pdf):
> "Push Particles Apart: Check all particle pairs is too slow! Use grid for speed up."
> "If two particles overlap, compute penetration depth and push them apart along their center line by half the overlap each."

## Whole-System Analysis

```
Current FLIP Pipeline:
┌─────────────────────────────────────────────────────────────────┐
│ 1. classify_cells()      → O(grid)                              │
│ 2. particles_to_grid()   → O(particles) - uses spatial hash     │
│ 3. store_old_velocities  → O(grid)                              │
│ 4. apply_gravity         → O(grid)                              │
│ 5. PRESSURE PROJECTION   → O(grid × iterations)                 │
│ 6. grid_to_particles()   → O(particles) - parallel              │
│ 7. apply_sediment_forces → O(particles) - parallel              │
│ 7b. [DISABLED] Clavet    → O(particles × neighbors) - EXPENSIVE │
│ 8. advect_particles()    → O(particles) - parallel, uses SDF    │
└─────────────────────────────────────────────────────────────────┘

We already have:
- Spatial hash (build_spatial_hash) - O(n) construction
- Linked-list per cell (cell_head, particle_next)
- SDF for solid collision
```

## Proposed Solution: FLIP-Native Push Apart

### Algorithm (from Matthias Müller)

```
for each particle i:
    for each neighbor j in same/adjacent cells:
        if distance(i, j) < 2 * particle_radius:
            overlap = 2 * particle_radius - distance
            direction = normalize(pos_i - pos_j)
            pos_i += direction * overlap * 0.5
            pos_j -= direction * overlap * 0.5
```

### Key Differences from Clavet

| Aspect | Clavet | Push Apart |
|--------|--------|------------|
| Kernel | SPH poly6/spiky | None - binary |
| Density calc | Yes (expensive) | No |
| Pressure calc | Yes (expensive) | No |
| Per-particle work | ~50 FLOPs × neighbors | ~10 FLOPs × overlaps |
| When to apply | Every frame | Only when overlapping |

### Integration Points

**Option A: Add to advect_particles()**
- After moving particle, check neighbors
- Push apart overlapping particles
- Pro: Single pass, reuses SDF collision
- Con: Sequential (but we can parallelize differently)

**Option B: Separate pass after advect**
- New function: `separate_particles_simple()`
- Uses existing spatial hash
- Pro: Clean separation of concerns
- Con: Extra pass

**Option C: Modify during P2G/G2P**
- Check density during grid transfer
- Pro: No extra pass
- Con: Mixes concerns

### Recommended: Option B

1. Keep existing `build_spatial_hash()`
2. Add new `push_particles_apart()` after advect
3. Simple overlap check, no SPH kernels
4. 1-2 iterations sufficient (Houdini default)

## Reseeding (Future Enhancement)

If push-apart isn't enough, add reseeding:

```rust
fn reseed_particles(&mut self) {
    const MIN_PER_CELL: usize = 3;
    const MAX_PER_CELL: usize = 12;

    for each fluid cell:
        count = particles in cell
        if count < MIN_PER_CELL:
            add particle at jittered position
        if count > MAX_PER_CELL:
            remove random particle
}
```

## Implementation Order

1. **First**: Simple push-apart (solves compression)
2. **If needed**: Add reseeding (solves gaps/clustering)
3. **If needed**: Tune iterations/strength

## Files to Modify

- `crates/sim/src/flip.rs`:
  - Add `push_particles_apart()` function
  - Call after `advect_particles()` in `update()`
  - Reuse `build_spatial_hash()` and `cell_head`/`particle_next`

## Expected Performance

| Method | Complexity | Est. Time @ 11k |
|--------|------------|-----------------|
| Clavet | O(n × neighbors × kernels) | 10ms |
| Push Apart | O(n × overlaps) | <1ms |

Push-apart only does work when particles actually overlap, and uses simple distance check instead of SPH kernel evaluation.

## Test Plan

1. Implement push_particles_apart()
2. Run with 10k+ particles
3. Verify: no flattening, no explosions
4. Measure: sim time should be ~3-4ms total (vs 13ms with Clavet)
5. Visual: particles should maintain volume

## Implementation Results (2025-12-20)

### Summary
Implementation complete and all tests passing (13/13).

### Key Changes Made

1. **`push_particles_apart()` in flip.rs**
   - Wall-aware asymmetric pushing (80/20 ratio when one particle near wall)
   - 2 iterations as recommended by Houdini FLIP
   - Uses existing spatial hash for O(n) neighbor lookup

2. **SDF Fix in grid.rs `compute_sdf()`**
   - Fixed critical bug: SDF values were not negative inside solid cells
   - Added second pass to negate distances inside solids
   - This creates a proper signed distance field

3. **Safety Net in `advect_particles()`**
   - Added direct `is_solid()` check after SDF-based collision
   - Catches edge cases where bilinear interpolation returns positive values
   - Pushes particles to nearest non-solid cell if stuck

### Test Results

| Test | Before Fix | After Fix |
|------|------------|-----------|
| Particles in solid cells | 232 | 0 |
| Overlap events (Test 3) | 2,027 | 7 |
| Min pair distance (Test 3) | 0.015 | 1.46 |
| Severe overlaps (Test 4) | 13 | 0 |
| Min separation (Test 4) | 0.09 | 2.49 |

### Root Cause of Solid Penetration

The SDF was computing **unsigned** distance to solids (always >= 0), not a proper signed distance field. When `sample_sdf()` interpolated near solid boundaries, particles inside solid cells could have positive SDF values due to blending with adjacent non-solid cells.

Fix: Added second pass in `compute_sdf()` to negate distances inside solid cells:
```rust
// Second pass: make distances NEGATIVE inside solid cells
for idx in 0..self.sdf.len() {
    if self.solid[idx] {
        self.sdf[idx] = -self.sdf[idx].max(cell_size * 0.5);
    }
}
```

## References

- [Houdini FLIP Solver](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html)
- [Matthias Müller - Ten Minute Physics FLIP](https://matthias-research.github.io/pages/tenMinutePhysics/18-flip.pdf)
- [Bridson - Fluid Simulation for Computer Graphics](https://www.cs.ubc.ca/~rbridson/fluidsimulation/)
