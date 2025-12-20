# Riffle Geometry Performance Impact

## Problem
Increasing riffle density (decreasing `riffle_spacing`) causes simulation slowdown.

## Observations

### Before SDF Implementation
- `spacing=30`: sim ~32-35ms for 696 particles
- `spacing=10`: sim ~60ms for 696 particles (nearly 2x slower)
- **Root cause theory**: More collision checks per particle

### After SDF + Parallelization
- `spacing=30`: sim ~26ms for 696 particles (improved from 45ms)
- `spacing=10`: sim ~76ms for 696 particles (still slower, but collision is now O(1))

## Analysis

See [collision-systems.md](../../architecture/collision-systems.md) for full system documentation.

### What Changed
1. **SDF Implementation**: Particle-solid collision is now O(1) per particle via precomputed signed distance field
2. **Parallelized advect_particles**: Uses rayon par_iter_mut

### Why Spacing Still Affects Performance
SDF makes collision detection riffle-independent. The remaining slowdown is likely from:

1. **Fluid dynamics behavior**:
   - More riffles = more turbulence
   - Particles bounce between confined spaces
   - Higher velocities require more substeps

2. **Pressure solver convergence**:
   - More complex solid boundaries = harder pressure solve
   - 10 iterations may not be sufficient
   - Higher divergence values

3. **Near-pressure (Clavet) work**:
   - More turbulence = more particle clustering
   - More neighbors per particle = O(n²) becomes expensive

## Implemented Solutions

### 1. SDF for O(1) Collision (DONE)
- `grid.compute_sdf()` precomputes distance field once
- `advect_particles()` uses `sample_sdf()` instead of cell checks
- See `grid.rs` lines 72-169

### 2. Parallelized Collision (DONE)
- `advect_particles()` now uses `par_iter_mut()`
- Runs collision detection across all CPU cores

## Profiling Results (Dec 2024)

Instrumented timing at ~11k particles (release build):

| Step | Time | % |
|------|------|---|
| classify | 200µs | 1% |
| P2G | 550µs | 4% |
| forces | 450µs | 3% |
| pressure | 1000µs | 7% |
| G2P | 600µs | 4% |
| **near+sed** | **10000µs** | **72%** |
| advect (SDF) | 550µs | 4% |

**The bottleneck is `apply_near_pressure` (Clavet SPH neighbor search), NOT collision detection.**

SDF collision (`advect`) is only 4% of time - the optimization worked!

### Why Near-Pressure Scales Poorly

`apply_near_pressure` is O(n × avg_neighbors):
- Uses spatial hash for neighbor lookup
- But with dense particle clustering (many riffles = turbulence = clustering)
- Average neighbors per particle increases significantly
- At 11k particles with high density, this dominates

### Optimization Options for Near-Pressure

1. **Skip more frames**: Currently runs every 2 frames. Try every 4 frames.
2. **Reduce search radius** (`near_pressure_h`): Fewer neighbors per particle
3. **Early-exit for sparse regions**: Skip particles with low neighbor count
4. **GPU compute**: Move to compute shader (major refactor)
5. **Hierarchical approach**: Multi-level neighbor search

## Next Steps

1. Experiment with near-pressure frame skip (every 4 frames)
2. Tune `near_pressure_h` parameter
3. Consider whether near-pressure is even needed (Clavet may be overkill)

## Files Changed
- `crates/sim/src/grid.rs` - SDF implementation
- `crates/sim/src/flip.rs` - advect_particles uses SDF
- `crates/sim/src/sluice.rs` - calls compute_sdf() after terrain setup
