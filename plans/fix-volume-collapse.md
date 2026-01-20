# Plan: Fix Volume Collapse Bug

## Status: ✅ RESOLVED

**Date:** 2026-01-19
**Solution:** Conservative cell expansion with 3+ neighbor threshold

## Problem Statement

The FLIP simulation loses ~84% of fluid volume during settling. When the emitter stops after filling the bucket with ~18,816 particles (expected 0.65m surface height), the fluid collapses to ~0.10m within 30 seconds.

## Root Cause Analysis

Standard FLIP implementations require cell marking expansion to prevent gaps in the fluid region. Without expansion, cells between particles are marked AIR, causing pressure solve failures and particle drift.

However, the standard "7 points per particle" approach (mark cell + all 6 neighbors) was **too aggressive** for our setup, causing the opposite problem - volume expansion (+27%).

## Solution: Conservative Cell Expansion

### Key Insight

The problem required a balanced approach:
- **No expansion:** 57% volume collapse
- **Any neighbor (1+):** 27% volume expansion + won't settle
- **2+ neighbors:** 10% volume expansion
- **3+ neighbors:** 3.5% collapse ✓ (within ±5% tolerance)

### Implementation

Modified `fluid_cell_expand_3d.wgsl` to require **3+ face-adjacent neighbors** with particles before marking an empty cell as FLUID:

```wgsl
// Minimum neighbors with particles to expand into empty cell
const MIN_NEIGHBORS_FOR_EXPANSION: u32 = 3u;

// Empty cell: only mark FLUID if surrounded by 3+ neighbors with particles
let neighbor_count = count_neighbors_with_particles(i32(i), i32(j), i32(k));
if (neighbor_count >= MIN_NEIGHBORS_FOR_EXPANSION) {
    cell_type[idx] = CELL_FLUID;
} else {
    cell_type[idx] = CELL_AIR;
}
```

### Test Results

```
Configuration     | Surface  | Error   | Settling
------------------|----------|---------|----------
No expansion      | 0.281m   | -57%    | ✓ Settles
1+ neighbors      | 0.830m   | +27%    | ✗ Won't settle
2+ neighbors      | 0.715m   | +10%    | ✓ Settling
3+ neighbors      | 0.628m   | -3.5%   | ✓ Settles (0.039 m/s)
```

**Final result:** Surface 0.628m (expected 0.651m) = **-3.5% error** (within ±5% tolerance)

## Files Modified

1. `crates/game/src/gpu/shaders/fluid_cell_expand_3d.wgsl` - Conservative expansion logic
2. `crates/game/src/gpu/shaders/velocity_extrapolate_3d.wgsl` - Velocity extrapolation (4 passes)
3. `crates/game/src/gpu/flip_3d.rs` - Pipeline integration

## Verification

```bash
cargo run --example bucket_fill_visual --release
```

Expected: Volume within ±5% after 30s settling. ✓

## References

- [Matthias Müller FLIP Tutorial](https://matthias-research.github.io/pages/tenMinutePhysics/18-flip.pdf)
- [Houdini FLIP Solver Docs](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html)
